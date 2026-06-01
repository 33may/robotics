#!/bin/bash
# One-off: bake UVA video_features into duck_cup_v020_all on the remote 5090.
# Output: eternalmay33/duck_cup_v020_all_uva (~41 GB fp16, gripper cam, L2 4x4, t_future=4)
export PYTHONPATH=/home/vbti/anton/robotics
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
source /home/vbti/anton/env/bin/activate
cd /home/vbti/anton/robotics || exit 1

echo "=== UVA BAKE START: $(date) ==="
python -m vbti.logic.dataset.add_video_features \
  --dataset eternalmay33/duck_cup_v020_all \
  --root /home/vbti/anton/data/eternalmay33/duck_cup_v020_all \
  --teacher /home/vbti/anton/data/uva_teacher_v020_150k \
  --layer siglip_output \
  --spatial-size 4 \
  --t-future 4 \
  --target-camera observation.images.gripper \
  --batch-size 32 \
  --dtype fp16 \
  --output /home/vbti/anton/data/eternalmay33/duck_cup_v020_all_uva
echo "=== UVA BAKE EXIT($?): $(date) ==="
