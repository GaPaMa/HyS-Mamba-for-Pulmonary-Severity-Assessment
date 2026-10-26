# Implementation and packaging notes

This package was assembled from the project's 16 attached coding files. It is a source release, not a newly trained or independently reproduced experiment.

## Preserved code

The following supplied implementations are unchanged: `HySMamba_regression.py`, `DINO_encoder_regression.py`, `tvimblock.py`, `trainer_new.py` (installed as `trainer.py`), `registry.py`, `configs.py`, `metrics.py`, `optimizers.py`, and `schedulers.py`. File paths are mapped in `source_manifest.json`.

## Packaging changes

- Restored the package paths expected by the supplied `age_estimation` imports and registry.
- Replaced active dataset paths in both experimental configs with `HYSMAMBA_DATA_ROOT`, defaulting to `data`. Removed obsolete commented experimental settings; active model/training values are preserved. Corrected the CT description to describe its actual linear CIP output.
- Imported Albumentations `ToTensorV2` explicitly and removed the local face-image demonstration from `data.py`. Other transform-building logic is retained. Albumentations 1.3.1 is specified for its original transform argument API.
- Removed the unconditional face-region-replacement import from the dataset module. That source file was not attached, and neither medical configuration uses it. Requesting face region replacement now raises a clear error.
- Made the unavailable optional center loss fail only when requested. The supplied composite MAE/MSE losses are unchanged.
- Replaced the machine-specific shell launcher with a portable Accelerate launcher and added seeding to the Python entry point.
- Added a runtime block-count/fallback check to training and evaluation. The original hybrid model's constructor and fallback code are unchanged.
- Added strict checkpoint evaluation and optional prediction export, plus a CUDA TViMBlock forward/backward check.
- Added a RALO 512-unit structural config reflecting the checkpoint head dimension disclosed in the manuscript; it does not certify that checkpoint's unavailable training settings.
- Extracted Figure 1 directly from the supplied revised PDF for the README.

## Reproducibility boundaries

The attached files do not include datasets, checkpoints, an exact environment lockfile, or the RALO expansion pipeline. The expanded RALO index and image order are required to use its fixed test indices. Row-level splitting after expansion does not guarantee patient-independent train/test partitions.

The original trainer and checkpoint-selection logic are retained. The package does not claim the documented manuscript scores were reproduced during preparation.

## Validation performed during preparation

Python compilation, shell syntax, Python/TOML configuration parsing, model-setting preservation, local import resolution, source checksums, RALO index partition behavior, evaluation CLI help, README local links, and archive contents were checked. The package's dependency-free `validate_source.py` can repeat the source checks.

This preparation environment has no PyTorch or CUDA runtime. Model forward passes, CUDA compilation, checkpoint loading and training were not run here. `scripts/check_environment.py` provides the target-environment CUDA check; a complete experiment additionally requires the original data and matching checkpoint.

