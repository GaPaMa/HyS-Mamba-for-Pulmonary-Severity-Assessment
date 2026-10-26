"""Evaluate an author-format checkpoint and optionally export predictions."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from age_estimation.configs import read_config
    from age_estimation.registry import DATASETS, METRICS, MODELS
    from age_estimation.utils.reproducibility import ensure_state_space
    if not torch.cuda.is_available():
        raise RuntimeError("The provided TViMBlock requires a CUDA GPU.")
    cfg = read_config(args.config_file)
    model = MODELS.build(cfg["model"])
    ensure_state_space(model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.cuda().eval()
    dataset = DATASETS.build(cfg[f"{args.split}_dataset"])
    loader = DataLoader(dataset, batch_size=cfg["trainer"]["batch_size"],
                        num_workers=cfg["trainer"].get("num_workers", 0),
                        pin_memory=True, shuffle=False)
    predictions, targets = [], []
    with torch.inference_mode():
        for images, labels in loader:
            outputs, _ = model(images.cuda(non_blocking=True))
            predictions.append(outputs.detach().float().cpu())
            targets.append(labels.float().reshape_as(outputs).cpu())
    if not predictions:
        raise ValueError("The requested dataset split is empty.")
    predictions, targets = torch.cat(predictions), torch.cat(targets)
    metric = METRICS.build(cfg["trainer"]["metrics"])
    results = {key: float(value) for key, value in metric(predictions, targets).items()}
    print(json.dumps({"num_samples": len(dataset), **results}, indent=2))
    if args.predictions:
        names = cfg["trainer"]["metrics"].get("dim_names") or ["CIP"]
        if len(names) != predictions.shape[1]:
            raise ValueError("Metric target names do not match checkpoint output dimensions.")
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row"] + [f"target_{name}" for name in names]
                            + [f"prediction_{name}" for name in names])
            for index, (label, pred) in enumerate(zip(targets.tolist(), predictions.tolist())):
                writer.writerow([index] + label + pred)


if __name__ == "__main__":
    main()

