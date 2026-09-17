#!/usr/bin/env bash
# Stage 2 for English: finetune the 7.48M student on Jenny's REAL recordings.
#
# The 7.48M model was distilled entirely from Kokoro-82M's own synthesis, so its ceiling is the
# teacher -- it can at best imitate a synthetic voice. Jenny is 26.45 h of one real speaker at
# 48 kHz, which is the same move that took the Arabic student from teacher-imitation to real
# audio, and the same DTW duration-transfer makes it possible.
#
# Warm start from runs/en7m_slm (the shipped model), lr 5e-5: this is a finetune, and the
# distillation run already put the decoder where it needs to be.
#
# Durations come from a CTC aligner trained on Jenny's own audio, NOT from DTW-transferring the
# teacher's. The DTW labels sat 2 frames late (offset_sweep.py peaks at -2 for them and at 0 for
# these) and training on them tripled free-running WER, 0.041 -> 0.145.
set -u
cd /opt/projects/math-gsk8/tts7
PY=/home/ahmed-wasfy/Documents/projects/kikiri-tts/.venv/bin/python

$PY -u train_student.py --steps 40000 --batch 8 --lr 5e-5 \
    --mel-weight 5 --sil-weight 100 --slm-weight 1 --slm-adv-weight 0.2 \
    --init-from ../tts5/runs/en7m_slm/last.pt \
    --index data/index_ctc.jsonl --bin dist/audio.i16.bin \
    --config config_student_wide.json --out runs/jenny_ctc --resume \
  || { echo "MARKER_FAILED $(date -Is)"; exit 1; }
echo "MARKER_JENNY_DONE $(date -Is)"
