"""Checks for the actual state-space blocks used by HyS-Mamba."""


def ensure_state_space(model):
    """Reject the author's convolution fallback when running a Mamba setting."""
    fallbacks = [name for name, module in model.named_modules()
                 if type(module).__name__ == "_SimpleVSS"]
    if fallbacks:
        raise RuntimeError(
            "TViMBlock was replaced by the convolution fallback at "
            + ", ".join(fallbacks)
            + ". Install a compatible selective_scan_cuda extension and run "
              "python scripts/check_environment.py before training/evaluation."
        )
    n_stages = len(getattr(model, "cn_channels", []))
    expected = n_stages * (
        int(getattr(model, "fusion_mode", "") == "damamba")
        + 2 * int(getattr(model, "projection_mode", "") == "biconvvss")
    )
    actual = sum(type(module).__name__ == "TViMBlock"
                 for module in model.modules())
    if actual != expected:
        raise RuntimeError(f"Expected {expected} TViMBlock modules; found {actual}.")
    return actual

