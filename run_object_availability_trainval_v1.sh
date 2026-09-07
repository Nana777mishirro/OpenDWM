#!/usr/bin/env bash
set -euo pipefail

OA_ROOT=/root/autodl-tmp/jttian-opendwm
OA_REPO="$OA_ROOT/OpenDWM"
OA_PYTHON="$OA_ROOT/envs/opendwm/bin/python3"
OA_DATA="$OA_ROOT/datasets/nuscenes-trainval"
OA_SMOKE_OUTPUT="$OA_ROOT/output/object-availability-trainval-full-smoke-v1"
OA_TRAIN_OUTPUT="$OA_ROOT/output/object-availability-formal-trainval-v1"
OA_EXPECTED_CAMERA_FILES=204894

mkdir -p "$OA_SMOKE_OUTPUT" "$OA_TRAIN_OUTPUT"
exec 9>"$OA_TRAIN_OUTPUT/launcher.lock"
if ! flock -n 9; then
    echo "Another object-availability Trainval launcher holds the lock."
    exit 1
fi

OA_CAMERA_FILES=$(find \
    "$OA_DATA/samples/CAM_BACK" \
    "$OA_DATA/samples/CAM_BACK_LEFT" \
    "$OA_DATA/samples/CAM_BACK_RIGHT" \
    "$OA_DATA/samples/CAM_FRONT" \
    "$OA_DATA/samples/CAM_FRONT_LEFT" \
    "$OA_DATA/samples/CAM_FRONT_RIGHT" \
    -type f | wc -l)
if [ "$OA_CAMERA_FILES" -ne "$OA_EXPECTED_CAMERA_FILES" ]; then
    echo "Camera file count mismatch: $OA_CAMERA_FILES != $OA_EXPECTED_CAMERA_FILES"
    exit 1
fi

for OA_REQUIRED_FILE in \
    "$OA_DATA/v1.0-trainval/scene.json" \
    "$OA_DATA/v1.0-trainval/sample.json" \
    "$OA_DATA/v1.0-trainval/sample_data.json" \
    "$OA_DATA/v1.0-trainval/sample_annotation.json"
do
    if [ ! -s "$OA_REQUIRED_FILE" ]; then
        echo "Missing metadata file: $OA_REQUIRED_FILE"
        exit 1
    fi
done

OA_READY_COUNT=0
while [ "$OA_READY_COUNT" -lt 3 ]
do
    OA_GPU_MEMORY=$(nvidia-smi -i 1 \
        --query-gpu=memory.used --format=csv,noheader,nounits)
    OA_GPU_UTIL=$(nvidia-smi -i 1 \
        --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    OA_GPU_MEMORY=${OA_GPU_MEMORY//[[:space:]]/}
    OA_GPU_UTIL=${OA_GPU_UTIL//[[:space:]]/}
    echo "GPU wait: memory=${OA_GPU_MEMORY}MiB util=${OA_GPU_UTIL}% ready=$OA_READY_COUNT/3"
    if [ "$OA_GPU_MEMORY" -lt 5000 ] && [ "$OA_GPU_UTIL" -lt 20 ]; then
        OA_READY_COUNT=$((OA_READY_COUNT + 1))
    else
        OA_READY_COUNT=0
    fi
    if [ "$OA_READY_COUNT" -lt 3 ]; then
        sleep 30
    fi
done

cd "$OA_REPO"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=1

echo "Starting one-batch full Trainval smoke at $(date -Is)."
"$OA_PYTHON" -u -m dwm.tools.object_availability_full_smoke \
    --config configs/object_availability_trainval_full_smoke_v1.json \
    --report-output "$OA_SMOKE_OUTPUT/full_report.json" \
    2>&1 | tee "$OA_SMOKE_OUTPUT/full_smoke.log"

echo "Starting formal Trainval training at $(date -Is)."
"$OA_PYTHON" -u -m dwm.train \
    --config-path configs/object_availability_trainval_formal_v1.json \
    --output-path "$OA_TRAIN_OUTPUT" \
    --log-steps 20 \
    --preview-steps 100000000 \
    --checkpointing-steps 1112 \
    --evaluation-steps 0 \
    2>&1 | tee "$OA_TRAIN_OUTPUT/train.log"

echo "Formal Trainval training completed at $(date -Is)."
sha256sum "$OA_TRAIN_OUTPUT"/checkpoints/*.pth
