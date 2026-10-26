import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from transformers import ViTModel

from lung_severity_assesmnt.registry import MODELS


def _gap(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)
    if x.dim() == 3:
        return x.mean(dim=1)
    return x


class ConvNeXtTimmFeatures(nn.Module):
    """ConvNeXt feature extractor using timm in features_only mode.

    We use timm because it easily exposes multi-scale feature maps (stages).
    """

    def __init__(self, timm_name: str = "convnext_tiny", pretrained: bool = True,
                 in_chans: int = 3, out_indices=(0, 1, 2, 3), checkpoint_path: str | None = None):
        super().__init__()
        self.model = timm.create_model(
            timm_name,
            pretrained=(pretrained and checkpoint_path is None),
            features_only=True,
            out_indices=tuple(out_indices),
            in_chans=in_chans,
        )
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            sd = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # tolerate prefixing
            sd = {k.replace("model.", ""): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)

        self.channels = self.model.feature_info.channels()
        self.out_indices = tuple(out_indices)
        
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)  # list of [B,C,H,W]


class CrossAttention2D(nn.Module):
    """Cross-attention: queries from a 2D map attend to K/V from another 2D map."""

    def __init__(self, q_dim: int, kv_dim: int, attn_dim: int = 256, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(q_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, attn_dim, bias=False)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(attn_dim, q_dim, bias=False)
        self.ln = nn.LayerNorm(q_dim)

    def forward(self, q_map: torch.Tensor, kv_map: torch.Tensor) -> torch.Tensor:
        """q_map: [B,Cq,Hq,Wq], kv_map: [B,Ckv,Hk,Wk]"""
        b, cq, hq, wq = q_map.shape
        b2, ckv, hk, wk = kv_map.shape
        assert b == b2

        q = q_map.flatten(2).transpose(1, 2)  # [B, Nq, Cq]
        kv = kv_map.flatten(2).transpose(1, 2)  # [B, Nk, Ckv]

        qn = self.ln(q)
        qh = self.q_proj(qn)
        kh = self.k_proj(kv)
        vh = self.v_proj(kv)

        ah, _ = self.attn(qh, kh, vh, need_weights=False)
        out = self.out_proj(ah) + q  # residual in q space
        out = out.transpose(1, 2).reshape(b, cq, hq, wq)
        return out


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class FAPM(nn.Module):
    """Fidelity-Aware Projection Module (paper's idea) for 2D maps.

    - shared context 1x1: D -> R
    - scale-specific 1x1: D -> R
    - context-conditioned modulation (FiLM-like)
    - refinement: DW-sep conv + SE + residual
    - final 1x1: R -> out_dim
    """

    def __init__(self, in_dim: int, rank: int = 256, out_dim: int = 256):
        super().__init__()
        self.rank = rank
        self.ctx = nn.Conv2d(in_dim, rank, kernel_size=1, bias=False)
        self.sp = nn.Conv2d(in_dim, rank, kernel_size=1, bias=False)

        # small generator: produce gamma/beta from context
        self.gen = nn.Sequential(
            nn.Conv2d(rank, rank, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(rank, 2 * rank, kernel_size=1),
        )

        self.dw = nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False)
        self.pw = nn.Conv2d(rank, rank, kernel_size=1, bias=False)
        self.se = SEBlock(rank)
        self.out = nn.Conv2d(rank, out_dim, kernel_size=1, bias=False)

        # self.bn = nn.BatchNorm2d(rank)
        # GroupNorm doesn't use running stats -> stable with small batches + DDP
        g = 32
        while rank % g != 0 and g > 1:
            g //= 2
        self.bn = nn.GroupNorm(num_groups=g, num_channels=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zctx = self.ctx(x)
        zsp = self.sp(x)
        gamma_beta = self.gen(zctx)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        zmod = gamma * zsp + beta

        y = self.dw(zmod)
        y = self.pw(y)
        y = self.bn(y)
        y = F.gelu(y)
        y = self.se(y)
        y = y + zmod
        return self.out(y)


@MODELS.register("dino_adapter_regressor")
class DinoAdapterRegressor(nn.Module):
    """Dino U-Net encoder idea adapted to regression.

    What matches your pipeline:
    - registered model name for configs
    - forward returns (pred, features)
    - supports freezing backbones + fine-tuning adapters/head
    """

    def __init__(
        self,
        convnext_kwargs: dict,
        vit_kwargs: dict,
        use_vit: bool = True,
        vit_layers: list[int] = [3, 6, 9, 11],
        out_indices: list[int] = [0, 1, 2, 3],
        attn_dim: int = 256,
        attn_heads: int = 8,
        fapm_rank: int = 256,
        fapm_out_dim: int = 256,
        head_hidden: int = 512,
        output_layers: list[int] = [1],
        freeze_vit: bool = True,
        freeze_convnext: bool = False,
    ):
        super().__init__()

        # -------- spatial branch (ConvNeXt) --------
        # expect timm-based convnext config (like your other configs)
        timm_name = convnext_kwargs.get("timm_name", "convnext_tiny")
        in_chans = convnext_kwargs.get("in_chans", 3)
        pretrained = convnext_kwargs.get("pretrained", True)
        ckpt = convnext_kwargs.get("pretrained_model_name_or_path", None)

        self.convnext = ConvNeXtTimmFeatures(
            timm_name=timm_name,
            pretrained=pretrained,
            in_chans=in_chans,
            out_indices=tuple(out_indices),
            checkpoint_path=ckpt,
        )
        self.cn_channels = self.convnext.channels
        self.out_indices = list(out_indices)

        # -------- semantic branch (ViT) --------
        # uses HF ViTModel. set output_hidden_states True in forward.
        # -------- semantic branch (ViT) --------
        self.use_vit = use_vit
        self.vit = ViTModel.from_pretrained(**vit_kwargs)
        self.vit_layers = list(vit_layers)
        self.vit_dim = self.vit.config.hidden_size

        # -------- adapter: cross-attn + FAPM per scale --------
        self.iblocks = nn.ModuleList()
        self.fapms = nn.ModuleList()
        for i, cdim in enumerate(self.cn_channels):
            self.iblocks.append(CrossAttention2D(q_dim=cdim, kv_dim=self.vit_dim, attn_dim=attn_dim, num_heads=attn_heads))
            self.fapms.append(FAPM(in_dim=cdim, rank=fapm_rank, out_dim=fapm_out_dim))

        # -------- regression head --------
        # in_head = fapm_out_dim * len(self.cn_channels)
        # layers: list[nn.Module] = [nn.LayerNorm(in_head), nn.Linear(in_head, head_hidden), nn.GELU()]
        # last = head_hidden
        # for j, out_f in enumerate(output_layers):
        #     layers.append(nn.Linear(last, out_f))
        #     if j < len(output_layers) - 1:
        #         layers.append(nn.ReLU())
        #         last = out_f
        # self.head = nn.Sequential(*layers)

    # -------- regression head (two-head multitask) --------
        self.out_dim = int(output_layers[-1]) if len(output_layers) > 0 else 1  # keeps your config style

        in_head = fapm_out_dim * len(self.cn_channels)

        self.center_feat_dim = 224
        self.center_proj = nn.Sequential(
            nn.LayerNorm(in_head),
            nn.Linear(in_head, self.center_feat_dim),
        )

        self.head_trunk = nn.Sequential(
            nn.LayerNorm(in_head),
            nn.Linear(in_head, head_hidden),
            nn.GELU(),
        )

        if self.out_dim == 2:
            self.head_ge = nn.Linear(head_hidden, 1)
            self.head_lo = nn.Linear(head_hidden, 1)
        else:
            self.head_single = nn.Linear(head_hidden, self.out_dim)
            # # optional center feature for your CenterLoss / multi_loss


        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
        if freeze_convnext:
            for p in self.convnext.parameters():
                p.requires_grad = False

    def _vit_to_map(self, tokens: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """tokens: [B, 1+N, C] -> map [B, C, Ht, Wt] resized to target_hw."""
        # remove CLS
        x = tokens[:, 1:, :]  # [B, N, C]
        b, n, c = x.shape
        s = int(math.sqrt(n))
        x = x.transpose(1, 2).reshape(b, c, s, s)
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # spatial multi-scale maps
        c_maps = self.convnext(pixel_values)  # list

        # semantic features
        if self.use_vit:
            vit_out = self.vit(pixel_values, output_hidden_states=True)
            hstates = vit_out.hidden_states  # tuple len = num_layers+1

            # pick viT layers; if fewer than scales, repeat last
            picked = [hstates[i] for i in self.vit_layers]
            if len(picked) < len(c_maps):
                picked = picked + [picked[-1]] * (len(c_maps) - len(picked))
            picked = picked[: len(c_maps)]
        else:
            picked = [None] * len(c_maps)

        fused_vecs = []
        for i, (cmap, vtoks) in enumerate(zip(c_maps, picked)):
            hw = (cmap.shape[-2], cmap.shape[-1])
            # vmap = self._vit_to_map(vtoks, hw)
            # enriched = self.iblocks[i](cmap, vmap)
            if self.use_vit:
                vmap = self._vit_to_map(vtoks, hw)
                enriched = self.iblocks[i](cmap, vmap)
            else:
                enriched = cmap
            proj = self.fapms[i](enriched)
            fused_vecs.append(_gap(proj))

        fused = torch.cat(fused_vecs, dim=1)
        center_feat = self.center_proj(fused)
        # out = self.head(fused)
        h = self.head_trunk(fused)

        if self.out_dim == 2:
            ge = self.head_ge(h)
            lo = self.head_lo(h)
            out = torch.cat([ge, lo], dim=-1)  # [B,2]
        else:
            out = self.head_single(h)          # [B,out_dim] or [B,1]

        # out = 8.0 * torch.sigmoid(out)
        return out, center_feat
    
    def freeze_all_vit(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze_vit_last_n(self, n: int = 4):
        # HF ViTModel blocks
        blocks = list(self.vit.encoder.layer)
        self.freeze_all_vit()
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True

    def param_groups(self, lr_low=1e-5, lr_high=2e-4, wd=0.05):
        # """Optional: two-tier LR (low for frozen-ish backbones, high for adapter/head)."""
        # low = []
        # high = []
        # for n, p in self.named_parameters():
        #     if not p.requires_grad:
        #         continue
        #     if n.startswith("vit") or n.startswith("convnext"):
        #         low.append(p)
        #     else:
        #         high.append(p)
        # return [
        #     {"params": low, "lr": lr_low, "weight_decay": wd},
        #     {"params": high, "lr": lr_high, "weight_decay": wd},
        # ]
        vit_params = list(self.vit.parameters())
        cn_params  = list(self.convnext.parameters())
        other_params = [p for n, p in self.named_parameters()
                        if not (n.startswith("vit") or n.startswith("convnext"))]

        return [
            {"params": vit_params,   "lr": lr_low,  "weight_decay": wd},
            {"params": cn_params,    "lr": lr_high, "weight_decay": wd},
            {"params": other_params, "lr": lr_high, "weight_decay": wd},
        ]