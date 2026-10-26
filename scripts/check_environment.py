"""Exercise the actual selective-scan CUDA forward and backward ABI."""
import sys


def main():
    try:
        import torch
        import selective_scan_cuda
        from lung_severity_assesmnt.models.tiny_vim.tvimblock import TViMBlock
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required by the supplied TViMBlock.")
        block = TViMBlock(hidden_dim=256, index=0).cuda()
        x = torch.randn(2, 256, 8, 8, device="cuda", requires_grad=True)
        y = block(x)
        if y.shape != x.shape or not torch.isfinite(y).all():
            raise RuntimeError("TViMBlock returned an invalid output.")
        y.square().mean().backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise RuntimeError("TViMBlock returned an invalid input gradient.")
        print(f"TViMBlock CUDA forward/backward passed (PyTorch {torch.__version__}).")
    except Exception as exc:
        print(f"State-space environment check failed: {exc}", file=sys.stderr)
        print("See the selective_scan_cuda installation notes in README.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

