"""Generate distillation dataset from hexgrad/Kokoro-82M teacher with two voices (1 Female, 1 Male).

Alternates sentences 50/50 between the female voice and male voice.
Dumps:
  1. audio.i16.bin: packed continuous int16 24 kHz PCM audio.
  2. index.jsonl: per-sentence metadata with phoneme ids, teacher-aligned durations,
     byte offsets, and speaker label ('female' / 'male').
  3. voice_female.pt & voice_male.pt: the exact style packs used during generation.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT / "kokoro_patched"))

from kokoro import KModel, KPipeline


SAMPLE_SENTENCES = [
    "Hello and welcome. This is a demonstration of multi-voice distillation.",
    "The quick brown fox jumps over the lazy dog in the bright morning sunshine.",
    "Artificial intelligence and speech synthesis have advanced dramatically in recent years.",
    "Reading books aloud is a wonderful way to improve pronunciation and fluency.",
    "She walked along the riverbank, watching the reflection of the clouds in the calm water.",
    "Modern text to speech models can run entirely on device without an internet connection.",
    "He explained the complex scientific concept with remarkable clarity and patience.",
    "Music and spoken poetry have accompanied human storytelling for thousands of years.",
    "The laboratory experiments yielded promising results that exceeded all initial expectations.",
    "Could you please repeat the instructions to make sure everyone understands the plan?",
    "Every journey begins with a single step forward into the unknown world.",
    "The astronomer pointed the telescope toward the distant spiral galaxy in the constellation.",
    "We should always preserve natural habitats and protect endangered species from extinction.",
    "Technology should empower people to communicate and share ideas across all languages.",
    "They spent the afternoon discussing philosophy, literature, and the beauty of mathematics.",
    "Cold winter winds swept across the open fields as the sun began to set.",
    "Education is the most powerful tool which you can use to change the world for the better.",
    "A gentle breeze rustled the autumn leaves as they fell softly to the forest floor.",
    "Please make sure your seatbelt is securely fastened before the flight departs.",
    "Innovation happens when curious minds ask questions that others have overlooked."
]


def main():
    ap = argparse.ArgumentParser(description="Generate distillation dataset with 2 voices")
    ap.add_argument("--texts", default=None,
                    help="Path to a text file (one sentence per line) or jsonl file with a 'text' field. "
                         "If omitted, uses a built-in sample set.")
    ap.add_argument("--female-voice", default="af_bella",
                    help="HuggingFace voice name or .pt path for the female speaker (default: af_bella)")
    ap.add_argument("--male-voice", default="am_adam",
                    help="HuggingFace voice name or .pt path for the male speaker (default: am_adam)")
    ap.add_argument("--out-dir", default="dist_twovoice",
                    help="Output directory for index.jsonl and audio.i16.bin (default: dist_twovoice)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Maximum number of sentences to process (optional)")
    ap.add_argument("--device", default=None,
                    help="Device to run inference on ('cuda' or 'cpu', default: auto)")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing teacher pipeline on {dev}...")

    # Initialize teacher pipeline & model
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=dev)
    teacher_model = pipe.model

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load and save style vectors locally
    print(f"[*] Loading female voice: {args.female_voice}")
    pack_female = pipe.load_voice(args.female_voice).cpu()
    print(f"[*] Loading male voice: {args.male_voice}")
    pack_male = pipe.load_voice(args.male_voice).cpu()

    torch.save(pack_female, out_path / "voice_female.pt")
    torch.save(pack_male, out_path / "voice_male.pt")

    voices = {
        "female": pack_female,
        "male": pack_male
    }

    # Gather sentences
    sentences = []
    if args.texts and os.path.exists(args.texts):
        with open(args.texts, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{") and line.endswith("}"):
                    try:
                        row = json.loads(line)
                        if "text" in row:
                            sentences.append(row["text"].strip())
                            continue
                    except json.JSONDecodeError:
                        pass
                sentences.append(line)
        print(f"[*] Loaded {len(sentences):,} sentences from {args.texts}")
    else:
        if args.texts:
            print(f"[!] Warning: texts file '{args.texts}' not found. Falling back to built-in sample sentences.")
        sentences = SAMPLE_SENTENCES
        print(f"[*] Using built-in sample set ({len(sentences)} sentences)")

    if args.max_samples:
        sentences = sentences[:args.max_samples]

    bin_file = out_path / "audio.i16.bin"
    idx_file = out_path / "index.jsonl"

    bin_fp = open(bin_file, "wb")
    idx_fp = open(idx_file, "w", encoding="utf-8")

    current_offset = 0
    generated_count = 0
    speaker_counts = {"female": 0, "male": 0}
    total_audio_sec = 0.0

    print(f"[*] Generating teacher speech into {out_path}...")
    for i, text in enumerate(sentences):
        # Alternate speakers 50/50
        speaker = "female" if (i % 2 == 0) else "male"
        voice_pack = voices[speaker]

        # G2P conversion
        phonemes_list, _ = pipe.g2p(text)
        token_ids = [v for v in (teacher_model.vocab.get(p) for p in phonemes_list) if v is not None]

        # Ensure valid length
        if not token_ids or len(token_ids) + 2 > teacher_model.context_length:
            continue

        # Format input_ids with leading and trailing 0 tokens
        input_ids = torch.LongTensor([[0, *token_ids, 0]]).to(dev)

        # Style reference slice matching sequence length
        ref = voice_pack[min(len(token_ids) - 1, voice_pack.shape[0] - 1)]
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        ref = ref.to(dev)

        with torch.no_grad():
            audio_tensor, pred_dur = teacher_model.forward_with_tokens(input_ids, ref, speed=1.0)

        audio_np = audio_tensor.squeeze().float().cpu().numpy()
        dur_list = pred_dur.squeeze().cpu().tolist()
        if isinstance(dur_list, int):
            dur_list = [dur_list]

        # Expected sample length: sum(dur) * 600
        expected_samples = int(sum(dur_list)) * 600
        if len(audio_np) < expected_samples:
            audio_np = np.pad(audio_np, (0, expected_samples - len(audio_np)))
        else:
            audio_np = audio_np[:expected_samples]

        # Quantize to int16
        audio_i16 = np.clip(audio_np * 32767.0, -32767.0, 32767.0).astype(np.int16)
        bin_fp.write(audio_i16.tobytes())

        record = {
            "ids": input_ids[0].cpu().tolist(),
            "dur": dur_list,
            "offset": current_offset,
            "samples": len(audio_i16),
            "lang": speaker,       # maps to PACKS[speaker] in train_student.py
            "speaker": speaker,
            "text": text
        }
        idx_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        current_offset += len(audio_i16)
        generated_count += 1
        speaker_counts[speaker] += 1
        total_audio_sec += len(audio_i16) / 24000.0

        if generated_count % 10 == 0 or generated_count == len(sentences):
            print(f"  [{generated_count}/{len(sentences)}] clips | "
                  f"Female: {speaker_counts['female']} | Male: {speaker_counts['male']} | "
                  f"Audio: {total_audio_sec:.1f}s", flush=True)

    bin_fp.close()
    idx_fp.close()

    print(f"\n[+] Dataset generation complete!")
    print(f"    Total clips: {generated_count}")
    print(f"    Speaker balance: {speaker_counts['female']} female, {speaker_counts['male']} male")
    print(f"    Audio duration: {total_audio_sec / 60.0:.2f} minutes ({total_audio_sec:.1f} s)")
    print(f"    Index: {idx_file}")
    print(f"    Binary: {bin_file}")
    print(f"    Style packs saved: {out_path / 'voice_female.pt'}, {out_path / 'voice_male.pt'}")


if __name__ == "__main__":
    main()
