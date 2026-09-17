#!/usr/bin/env bash
set -e

echo "=== 1. Checking GPU ==="
nvidia-smi

echo "=== 2. Installing Dependencies ==="
pip install -r requirements.txt
python -m spacy download en_core_web_sm || true

echo "=== 3. Fetching Sentences & Generating Teacher Dataset ==="
# Fetch 3000 sentences for real training (or supply your own sentences.txt)
if [ ! -f "sentences.txt" ]; then
    echo "Fetching 3000 clean sentences..."
    python training/fetch_sentences.py --count 3000 --out sentences.txt
fi

python training/generate_teacher_dataset.py \
    --texts sentences.txt \
    --female-voice af_bella \
    --male-voice am_adam \
    --out-dir dist_twovoice

echo "=== 4. Training 7.48M Student Model on GPU ==="
# T4 GPU handles batch size 16 easily at ~0.15s/step
python training/train_student.py \
    --index dist_twovoice/index.jsonl \
    --bin dist_twovoice/audio.i16.bin \
    --config config.json \
    --two-voices \
    --batch 16 \
    --steps 20000 \
    --mel-weight 5.0 \
    --sil-weight 100.0 \
    --lr 2e-4 \
    --lr-decay \
    --out runs/twovoice_student

echo "=== 5. Packaging Complete Model Bundle ==="
# Zips student.pth + voice_female.pt + voice_male.pt + config.json into one download
zip -j twovoice_model.zip \
    runs/twovoice_student/student.pth \
    runs/twovoice_student/voice_female.pt \
    runs/twovoice_student/voice_male.pt \
    runs/twovoice_student/config.json

echo "=== Training Complete! ==="
echo "Download 'twovoice_model.zip' which contains model weights AND both voice packs."
