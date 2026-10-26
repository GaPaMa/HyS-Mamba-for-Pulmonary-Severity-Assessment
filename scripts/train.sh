#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
config_file="${1:-configs/dino_ViM_CT.py}"
if (( $# > 0 )); then shift; fi
python scripts/check_environment.py
gpu_count="$(python -c 'import torch; print(torch.cuda.device_count())')"
launch_args=(--num_processes="$gpu_count" --num_machines=1 --mixed_precision=no --dynamo_backend=no)
if (( gpu_count > 1 )); then launch_args+=(--multi_gpu); fi
if [[ -n "${MASTER_PORT:-}" ]]; then launch_args+=(--main_process_port="$MASTER_PORT"); fi
accelerate launch "${launch_args[@]}" scripts/train.py --config_file "$config_file" "$@"

