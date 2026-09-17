"""Jenny 48 kHz raw -> trimmed 24 kHz flac + manifest.

Two preprocessing steps the dataset card explicitly asks for: the audio is "raw from the
microphone, not trimmed", with seconds of leading silence and an occasional key-press knock
where the speaker hit record. That matters more for us than for most pipelines, because the
student has an explicit silence loss -- untrimmed leading silence would teach it to emit
several seconds of nothing before speaking.

Trim thresholds are deliberately conservative (top_db=35): over-trimming clips word onsets,
which is worse than leaving a little silence, and the forced aligner can absorb small leading
pauses but not a missing first phoneme.
"""
from __future__ import annotations
import glob, io, json, os, sys, time
import numpy as np, soundfile as sf, soxr, librosa

SR = 24000
OUT = "data/wavs"
PAT = os.path.expanduser("~/.cache/huggingface/hub/datasets--reach-vb--jenny_tts_dataset/"
                         "snapshots/*/**/*.parquet")


def main():
    import pyarrow.parquet as pq
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(PAT, recursive=True))
    mf = open("data/jenny.jsonl", "w", encoding="utf-8")
    kept = dropped = 0; secs = 0.0; trimmed = 0.0
    t0 = time.time()
    for si, shard in enumerate(files):
        for j, r in enumerate(pq.ParquetFile(shard).read().to_pylist()):
            txt = (r.get("transcription") or "").strip()
            au = r.get("audio")
            if not txt or not isinstance(au, dict) or not au.get("bytes"):
                dropped += 1; continue
            try:
                w, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
            except Exception:
                dropped += 1; continue
            if w.ndim > 1:
                w = w.mean(axis=1)                      # card notes 2ch from one mic
            raw = len(w) / sr
            w, _ = librosa.effects.trim(w, top_db=35)
            if sr != SR:
                w = soxr.resample(w, sr, SR)
            dur = len(w) / SR
            if not (0.5 <= dur <= 30.0):
                dropped += 1; continue
            p = f"{OUT}/jenny_{si:02d}_{j:05d}.flac"
            sf.write(p, w, SR, format="FLAC")
            mf.write(json.dumps({"audio": os.path.abspath(p), "text": txt,
                                 "duration": round(dur, 3)}, ensure_ascii=False) + "\n")
            kept += 1; secs += dur; trimmed += raw - dur
        mf.flush()
        print(f"  [{si+1}/{len(files)}] kept {kept:,}  {secs/3600:.2f} h  "
              f"trimmed {trimmed/60:.1f} min of silence  ({(time.time()-t0)/60:.1f} min)",
              flush=True)
    mf.close()
    print(f"[+] {kept:,} clips, {secs/3600:.2f} h, dropped {dropped}, "
          f"removed {trimmed/60:.1f} min of leading/trailing silence", flush=True)


if __name__ == "__main__":
    main()
