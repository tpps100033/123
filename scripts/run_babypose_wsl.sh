#!/usr/bin/env bash
set -euo pipefail
echo "BabyPose is archived and excluded from the supported COCO/SyRIP/SVIRO benchmark. Use scripts/run_all_research.sh." >&2
exit 2

cd "$(dirname "$0")/.."
venv_root="${INFANTPOSE_CACHE_ROOT:-$HOME/.local/share/infantpose}/venvs"

python3 scripts/prepare_babypose.py

has_complete_run() {
    python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

dataset = "babypose_test160_limb12"
expected = json.loads((Path("02_datasets/canonical") / dataset / "dataset_manifest.json").read_text())["index_sha256"]
root = Path("03_model_raw_outputs") / sys.argv[1] / dataset / "fair_reference"
for path in root.glob("*/run_manifest.json"):
    run = json.loads(path.read_text())
    if run.get("metric_complete") and run.get("counts", {}).get("input") == 160 and run.get("data_handoff_hash") == expected:
        print(f"[BabyPose] reuse complete run: {path}")
        raise SystemExit(0)
raise SystemExit(1)
PY
}

models=(mediapipe_pose_heavy movenet_thunder rtmpose_m yolo11x_pose sapiens_1b_coco openpose_coco18)
environments=(mediapipe movenet rtmpose yolo sapiens openpose)
failed=0
for i in "${!models[@]}"; do
    model="${models[$i]}"
    python="${venv_root}/${environments[$i]}/bin/python"
    if has_complete_run "$model"; then
        continue
    fi
    echo "[BabyPose] validate ${model}"
    if ! "$python" scripts/validate_model.py --model "$model" --dataset babypose_test160_limb12; then
        echo "[BabyPose] validation failed for ${model}; continuing with the next model" >&2
        failed=1
        continue
    fi
    echo "[BabyPose] evaluate ${model} on 160 images"
    if ! "$python" scripts/run_research.py --model "$model" --dataset babypose_test160_limb12; then
        echo "[BabyPose] evaluation failed for ${model}" >&2
        failed=1
        if "$python" -c 'import json,sys; from pathlib import Path; p=sorted((Path("03_model_raw_outputs")/sys.argv[1]/"babypose_test160_limb12"/"fair_reference").glob("*/run_manifest.json")); raise SystemExit(0 if p and json.loads(p[-1].read_text()).get("thermal_stop") else 1)' "$model"; then
            echo "[BabyPose] thermal stop recorded; halting remaining GPU work" >&2
            exit 2
        fi
    fi
done

python3 scripts/compare_babypose.py
exit "$failed"
