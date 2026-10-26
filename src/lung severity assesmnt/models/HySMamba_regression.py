
import math
import os
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

# DropPath import for timm compatibility
try:
    from timm.models.layers import DropPath
except Exception:
    from timm.layers import DropPath

from transformers import ViTModel

from age_estimation.registry import MODELS


# --------------------- small helpers ---------------------
def _gap(x: torch.Tensor) -> torch.Tensor:
    """Global average pooling for maps or token sequences."""
    if x.dim() == 4:
        return F.adaptive_avg_pool2d(x, 1).flatten(1)
    if x.dim() == 3:
        return x.mean(dim=1)
    return x


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for NCHW maps (stable alternative to BN for small batch)."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(1, channels, eps=eps, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gn(x)


def _try_import_tvimblock():
    """
    Prefer your Mamba/VSS implementation (TViMBlock).
    Fallback to a safe conv-MLP block if the CUDA op isn't available.
    """
    try:
        from age_estimation.models.tiny_vim.tvimblock import TViMBlock  # your repo path
        return TViMBlock, True
    except Exception:
        return None, False


class _SimpleVSS(nn.Module):
    """Fallback block when TViMBlock / selective_scan_cuda isn't available."""
    def __init__(self, dim: int, drop: float = 0.0):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim * 2, 1)
        self.dw  = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * 2, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.pw1(h)
        h = self.dw(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.pw2(h)
        return x + h


def _make_vss_block(dim: int, vss_kwargs: Optional[Dict[str, Any]] = None, index: Optional[int] = None) -> nn.Module:
    TViMBlock, ok = _try_import_tvimblock()
    kwargs = dict(vss_kwargs or {})
    if index is not None:
        kwargs["index"] = int(index)

    if ok:
        # TViMBlock signature varies in forks; keep kwargs optional and minimal.
        try:
            return TViMBlock(dim, **kwargs)
        except TypeError:
            # Some forks do not accept index; retry without it before fallback.
            kwargs_no_index = dict(kwargs)
            kwargs_no_index.pop("index", None)
            try:
                return TViMBlock(dim, **kwargs_no_index)
            except TypeError:
                # if ctor mismatch, fall back gracefully
                return _SimpleVSS(dim, drop=float(kwargs.get("drop", 0.0)))

    return _SimpleVSS(dim, drop=float(kwargs.get("drop", 0.0)))


class ConvNeXtTimmFeatures(nn.Module):
    """ConvNeXt feature extractor using timm in features_only mode."""

    def __init__(
        self,
        timm_name: str = "convnext_tiny",
        pretrained: bool = True,
        in_chans: int = 3,
        out_indices=(0, 1, 2, 3),
        checkpoint_path: str | None = None,
    ):
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
            sd = {k.replace("model.", ""): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)

        self.channels = self.model.feature_info.channels()
        self.out_indices = tuple(out_indices)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.model(x)  # list of [B,C,H,W]


# --------------------- CoAG: conv <-> vit alignment gating ---------------------
class CoAG(nn.Module):
    """
    Contextual Alignment Gating between spatial conv features C_i and semantic ViT features F_i.

    - Computes global descriptors from both branches.
    - Produces channel-wise gates for each branch (sigmoid).
    - Applies gentle scaling (0.5..1.5) to avoid shutting channels off.
    """
    def __init__(self, c_dim: int, v_dim: int, hidden: int = 256, drop: float = 0.0):
        super().__init__()
        hidden = int(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(c_dim + v_dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, c_dim + v_dim),
        )

    def forward(self, c_map: torch.Tensor, v_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, cc, _, _ = c_map.shape
        _, cv, _, _ = v_map.shape

        c_g = _gap(c_map)  # [B, cc]
        v_g = _gap(v_map)  # [B, cv]
        g = self.mlp(torch.cat([c_g, v_g], dim=1))
        gc, gv = g[:, :cc], g[:, cc:]

        # 0.5..1.5 scaling
        sc = 0.5 + torch.sigmoid(gc)
        sv = 0.5 + torch.sigmoid(gv)

        c_map = c_map * sc.view(b, cc, 1, 1)
        v_map = v_map * sv.view(b, cv, 1, 1)
        return c_map, v_map


# --------------------- DaMamba: Mamba/VSS fusion block ---------------------
class DaMambaFusion(nn.Module):
    """
    Replace the paper's (deformable) interaction block with a Mamba/VSS-style fusion:

      C_i, F_i -> (proj + fuse) -> VSS (2D) -> proj back -> residual into C_i

    This is "best-for-regression" biased:
    - Stable norms (LN2d)
    - Residual + DropPath
    - Minimal dependence on attention (works better with small data)
    """
    def __init__(
        self,
        c_dim: int,
        v_dim: int,
        inner_dim: int = 256,
        drop: float = 0.0,
        drop_path: float = 0.0,
        vss_kwargs: Optional[Dict[str, Any]] = None,
        vss_index: int = 0,
    ):
        super().__init__()
        self.c_norm = LayerNorm2d(c_dim)
        self.v_norm = LayerNorm2d(v_dim)

        self.c_proj = nn.Conv2d(c_dim, inner_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(v_dim, inner_dim, 1, bias=False)

        # fuse: concat then 1x1 to inner_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(inner_dim * 2, inner_dim, 1, bias=False),
            nn.GELU(),
        )

        self.vss = _make_vss_block(inner_dim, vss_kwargs=vss_kwargs, index=vss_index)
        self.out = nn.Conv2d(inner_dim, c_dim, 1, bias=False)

        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, c_map: torch.Tensor, v_map: torch.Tensor) -> torch.Tensor:
        # normalize each branch before projection
        c = self.c_proj(self.c_norm(c_map))
        v = self.v_proj(self.v_norm(v_map))
        z = self.fuse(torch.cat([c, v], dim=1))
        z = self.vss(z)
        z = self.drop(z)
        z = self.out(z)
        return c_map + self.drop_path(z)


class ConvFusion(nn.Module):
    """Lightweight non-Mamba fusion baseline for DaMamba replacement ablations."""
    def __init__(
        self,
        c_dim: int,
        v_dim: int,
        inner_dim: int = 256,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.c_norm = LayerNorm2d(c_dim)
        self.v_norm = LayerNorm2d(v_dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(c_dim + v_dim, inner_dim, 1, bias=False),
            LayerNorm2d(inner_dim),
            nn.GELU(),
            nn.Conv2d(inner_dim, inner_dim, 3, padding=1, groups=inner_dim, bias=False),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(inner_dim, c_dim, 1, bias=False),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, c_map: torch.Tensor, v_map: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.c_norm(c_map), self.v_norm(v_map)], dim=1)
        return c_map + self.drop_path(self.fuse(z))


class IdentityFusion(nn.Module):
    """No semantic-to-spatial fusion; keeps ConvNeXt map unchanged."""
    def forward(self, c_map: torch.Tensor, v_map: torch.Tensor) -> torch.Tensor:
        return c_map


# --------------------- BiConvVSS: projection/refinement replacing FAPM ---------------------
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


class BiConvVSS(nn.Module):
    """
    Replace FAPM with a bidirectional Conv + VSS projection.

    Steps:
    - 1x1 projection to rank
    - local conv refinement (DW)
    - VSS forward
    - VSS "reverse" via spatial flip (cheap bidirectionality)
    - fuse + SE + 1x1 to out_dim
    """
    def __init__(
        self,
        in_dim: int,
        rank: int = 256,
        out_dim: int = 256,
        drop: float = 0.0,
        vss_kwargs: Optional[Dict[str, Any]] = None,
        vss_index: int = 0,
    ):
        super().__init__()
        self.in_norm = LayerNorm2d(in_dim)

        self.in_proj = nn.Conv2d(in_dim, rank, 1, bias=False)
        self.dw = nn.Conv2d(rank, rank, 3, padding=1, groups=rank, bias=False)
        self.pw = nn.Conv2d(rank, rank, 1, bias=False)

        self.vss_fwd = _make_vss_block(rank, vss_kwargs=vss_kwargs, index=vss_index)
        self.vss_rev = _make_vss_block(rank, vss_kwargs=vss_kwargs, index=vss_index)

        self.se = SEBlock(rank)
        self.drop = nn.Dropout(drop)
        self.out = nn.Conv2d(rank, out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(self.in_norm(x))
        z = F.gelu(self.pw(self.dw(z)))

        y_fwd = self.vss_fwd(z)
        # reverse pass (flip W), then flip back
        z_rev = torch.flip(z, dims=[-1])
        y_rev = torch.flip(self.vss_rev(z_rev), dims=[-1])

        y = y_fwd + y_rev + z
        y = self.se(y)
        y = self.drop(y)
        return self.out(y)


class FAPMProjection(nn.Module):
    """Original DINO adapter-style projection baseline for BiConvVSS ablations."""
    def __init__(self, in_dim: int, rank: int = 256, out_dim: int = 256):
        super().__init__()
        self.ctx = nn.Conv2d(in_dim, rank, kernel_size=1, bias=False)
        self.sp = nn.Conv2d(in_dim, rank, kernel_size=1, bias=False)
        self.gen = nn.Sequential(
            nn.Conv2d(rank, rank, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(rank, 2 * rank, kernel_size=1),
        )
        self.dw = nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank, bias=False)
        self.pw = nn.Conv2d(rank, rank, kernel_size=1, bias=False)
        self.se = SEBlock(rank)
        self.out = nn.Conv2d(rank, out_dim, kernel_size=1, bias=False)

        groups = 32
        while rank % groups != 0 and groups > 1:
            groups //= 2
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zctx = self.ctx(x)
        zsp = self.sp(x)
        gamma, beta = torch.chunk(self.gen(zctx), 2, dim=1)
        zmod = gamma * zsp + beta

        y = self.dw(zmod)
        y = self.pw(y)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.se(y)
        return self.out(y + zmod)


class SimpleProjection(nn.Module):
    """Minimal projection baseline without BiConvVSS, bidirectionality, or SE refinement."""
    def __init__(self, in_dim: int, rank: int = 256, out_dim: int = 256, drop: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            LayerNorm2d(in_dim),
            nn.Conv2d(in_dim, rank, 1, bias=False),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(rank, out_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# --------------------- The model: Dino-style multi-scale encoder -> regression ---------------------
@MODELS.register("dino_coag_damamba_biconvvss_regressor")
class DinoCoAGDaMambaBiConvVSSRegressor(nn.Module):
    """
    A Dino U-Net inspired encoder customized for regression, using the components from your TODO slide:

      - 4x CoAG between C_i and ViT features
      - 4x DaMamba fusion blocks (replace interaction blocks)
      - BiConvVSS projection blocks (replace FAPM)

    Notes:
    - Still uses HF ViTModel (unless you swap it for DINOv3 in your codebase)
    - Designed to plug into trainer_new.py: returns (pred, features)
    """

    def __init__(
        self,
        convnext_kwargs: Dict[str, Any],
        vit_kwargs: Dict[str, Any],
        vit_layers: List[int] = [3, 6, 9, 11],
        out_indices: List[int] = [0, 1, 2, 3],
        # fusion / vss
        inner_dim: int = 256,
        vss_kwargs: Optional[Dict[str, Any]] = None,
        drop: float = 0.0,
        drop_path: float = 0.0,
        # projection
        proj_rank: int = 256,
        proj_out_dim: int = 256,
        # head
        head_hidden: int = 512,
        head_drop: float = 0.1,
        output_layers: List[int] = [1],
        output_activation: Optional[str] = None,
        output_scale: float = 1.0,
        # center feature
        center_dim: int = 224,
        # freezing
        freeze_vit: bool = True,
        freeze_convnext: bool = False,
        # ablations
        use_coag: bool = True,
        fusion_mode: str = "damamba",
        projection_mode: str = "biconvvss",
    ):
        super().__init__()
        self.use_coag = bool(use_coag)
        self.fusion_mode = str(fusion_mode).lower()
        self.projection_mode = str(projection_mode).lower()
        if self.fusion_mode not in ("damamba", "conv", "identity", "none"):
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}. "
                "Use 'damamba', 'conv', 'identity', or 'none'."
            )
        if self.projection_mode not in ("biconvvss", "fapm", "simple"):
            raise ValueError(
                f"Unsupported projection_mode={projection_mode!r}. "
                "Use 'biconvvss', 'fapm', or 'simple'."
            )

        # -------- spatial branch (ConvNeXt) --------
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
        self.cn_channels = list(self.convnext.channels)
        self.out_indices = list(out_indices)

        # -------- semantic branch (ViT) --------
        self.vit = ViTModel.from_pretrained(**vit_kwargs)
        self.vit_layers = list(vit_layers)
        self.vit_dim = int(self.vit.config.hidden_size)

        # -------- CoAG + DaMamba + BiConvVSS per scale --------
        self.coags = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.projs = nn.ModuleList()

        # distribute droppath across stages
        dp = [x.item() for x in torch.linspace(0, float(drop_path), steps=len(self.cn_channels))]

        for i, cdim in enumerate(self.cn_channels):
            self.coags.append(CoAG(c_dim=cdim, v_dim=self.vit_dim, hidden=max(128, inner_dim), drop=drop))
            if self.fusion_mode == "damamba":
                fusion = DaMambaFusion(
                    c_dim=cdim,
                    v_dim=self.vit_dim,
                    inner_dim=inner_dim,
                    drop=drop,
                    drop_path=dp[i],
                    vss_kwargs=vss_kwargs,
                    vss_index=i,
                )
            elif self.fusion_mode == "conv":
                fusion = ConvFusion(
                    c_dim=cdim,
                    v_dim=self.vit_dim,
                    inner_dim=inner_dim,
                    drop=drop,
                    drop_path=dp[i],
                )
            else:
                fusion = IdentityFusion()
            self.fusions.append(fusion)

            if self.projection_mode == "biconvvss":
                proj = BiConvVSS(
                    in_dim=cdim,
                    rank=proj_rank,
                    out_dim=proj_out_dim,
                    drop=drop,
                    vss_kwargs=vss_kwargs,
                    vss_index=i,
                )
            elif self.projection_mode == "fapm":
                proj = FAPMProjection(in_dim=cdim, rank=proj_rank, out_dim=proj_out_dim)
            else:
                proj = SimpleProjection(in_dim=cdim, rank=proj_rank, out_dim=proj_out_dim, drop=drop)
            self.projs.append(proj)

        # -------- regression head --------
        self.out_dim = int(output_layers[-1]) if len(output_layers) > 0 else 1
        self.output_activation = None if output_activation is None else str(output_activation).lower()
        self.output_scale = float(output_scale)
        if self.output_activation not in (None, "sigmoid"):
            raise ValueError(
                f"Unsupported output_activation={output_activation!r}. "
                "Use None or 'sigmoid'."
            )
        in_head = proj_out_dim * len(self.cn_channels)

        self.head_trunk = nn.Sequential(
            nn.LayerNorm(in_head),
            nn.Linear(in_head, head_hidden),
            nn.GELU(),
            nn.Dropout(head_drop),
        )

        if self.out_dim == 2:
            self.head_ge = nn.Linear(head_hidden, 1)
            self.head_lo = nn.Linear(head_hidden, 1)
        else:
            self.head_single = nn.Linear(head_hidden, self.out_dim)

        # center feature for RegressionCenterLoss / multi_loss
        self.center_dim = int(center_dim) if center_dim is not None else 0
        if self.center_dim > 0:
            self.center_proj = nn.Sequential(
                nn.LayerNorm(in_head),
                nn.Linear(in_head, self.center_dim),
            )
        else:
            self.center_proj = None

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
        if freeze_convnext:
            for p in self.convnext.parameters():
                p.requires_grad = False

    # ---------- utilities ----------
    def _vit_to_map(self, tokens: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """
        tokens: [B, 1+N, C] -> map [B, C, Ht, Wt] resized to target_hw.
        Assumes square patch grid (typical ViT).
        """
        x = tokens[:, 1:, :]  # drop CLS
        b, n, c = x.shape
        s = int(math.sqrt(n))
        if s * s != n:
            # last resort: treat as 1D and reshape to closest square
            s = int(round(math.sqrt(n)))
        x = x[:, : s * s, :].transpose(1, 2).reshape(b, c, s, s)
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x

    def _apply_output_activation(self, out: torch.Tensor) -> torch.Tensor:
        if self.output_activation is None:
            return out
        if self.output_activation == "sigmoid":
            return self.output_scale * torch.sigmoid(out)
        raise RuntimeError(f"Unhandled output_activation={self.output_activation!r}")

    # ---------- forward ----------
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # spatial multi-scale maps
        c_maps = self.convnext(pixel_values)  # list of [B,C,H,W]

        # semantic features (keep hidden states for multi-scale tapping)
        vit_out = self.vit(pixel_values, output_hidden_states=True)
        hstates = vit_out.hidden_states  # tuple len = num_layers+1

        # pick ViT layers; if fewer than scales, repeat last
        picked = [hstates[i] for i in self.vit_layers]
        if len(picked) < len(c_maps):
            picked = picked + [picked[-1]] * (len(c_maps) - len(picked))
        picked = picked[: len(c_maps)]

        pooled = []
        for i, (cmap, vtoks) in enumerate(zip(c_maps, picked)):
            hw = (cmap.shape[-2], cmap.shape[-1])
            vmap = self._vit_to_map(vtoks, hw)

            if self.use_coag:
                cmap_g, vmap_g = self.coags[i](cmap, vmap)
            else:
                cmap_g, vmap_g = cmap, vmap
            enriched = self.fusions[i](cmap_g, vmap_g)
            proj = self.projs[i](enriched)
            pooled.append(_gap(proj))

        fused = torch.cat(pooled, dim=1)  # [B, proj_out_dim * 4]

        # center features for center loss (if used)
        features = self.center_proj(fused) if self.center_proj is not None else fused

        h = self.head_trunk(fused)
        if self.out_dim == 2:
            ge = self.head_ge(h)
            lo = self.head_lo(h)
            out = torch.cat([ge, lo], dim=-1)
        else:
            out = self.head_single(h)
        out = self._apply_output_activation(out)

        return out, features

    # ---------- fine-tuning helpers ----------
    def freeze_all_vit(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze_vit_last_n(self, n: int = 4):
        blocks = list(self.vit.encoder.layer)
        self.freeze_all_vit()
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True

    def param_groups(self, lr_low=1e-5, lr_high=2e-4, wd=0.05):
        """
        Trainer-compatible parameter groups.

        - lr_low: typically for the (mostly frozen) ViT / foundation encoder
        - lr_high: for ConvNeXt + adapters/projections + head
        """
        vit_params = [p for p in self.vit.parameters() if p.requires_grad]
        cn_params  = [p for p in self.convnext.parameters() if p.requires_grad]
        other_params = [
            p for n, p in self.named_parameters()
            if p.requires_grad and (not n.startswith("vit")) and (not n.startswith("convnext"))
        ]
        groups = []
        if vit_params:
            groups.append({"params": vit_params, "lr": lr_low, "weight_decay": wd})
        if cn_params:
            groups.append({"params": cn_params, "lr": lr_high, "weight_decay": wd})
        if other_params:
            groups.append({"params": other_params, "lr": lr_high, "weight_decay": wd})
        return groups
