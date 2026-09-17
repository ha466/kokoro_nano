"""Viterbi forced alignment of Jenny's audio with the CTC aligner -> per-phoneme durations.

Frames are the student's own 40 Hz grid, so a duration written here means exactly what the
trainer will read. Blank frames between two phonemes are given to the PRECEDING phoneme, which
keeps the durations contiguous and total-preserving: the decoder consumes sum(dur) frames of
audio and any gap in the labels would silently shift everything after it.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, soundfile as sf, torch
import torch.nn.functional as F
import torchaudio.functional as AF

from train_aligner import Aligner, melspec, load_rows, BLANK, HOP, SR


PAUSE = None          # set in main(): ids whose phoneme is silence-like


def segment(ids, dur, seg_max, seg_min, pause):
    """Cut one aligned recording into training segments at the pauses the aligner found.

    Greedy: walk forward to seg_max frames, then step back to the longest pause token in that
    window and cut there, so a segment never begins or ends mid-word. With seg_max = 0 the
    recording is returned whole."""
    total = int(sum(dur))
    if not seg_max or total <= seg_max:
        return [{"start": 0, "ids": list(ids), "dur": list(dur), "frames": total}]
    segs, i, pos = [], 0, 0
    n = len(dur)
    while i < n:
        acc, j, best, best_d = 0, i, -1, 0
        while j < n and acc + dur[j] <= seg_max:
            acc += dur[j]
            if ids[j] in pause and dur[j] >= best_d:
                best, best_d = j, dur[j]
            j += 1
        if j >= n:                                   # tail
            cut = n
        elif best > i:                               # cut at the widest pause in the window
            cut = best + 1
        else:                                        # no pause found; hard cut
            cut = max(j, i + 1)
        sd = list(dur[i:cut]); si = list(ids[i:cut])
        f = int(sum(sd))
        if f >= seg_min:
            segs.append({"start": pos, "ids": si, "dur": sd, "frames": f})
        pos += f
        i = cut
    return segs


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/aligner/last.pt")
    ap.add_argument("--max-frames", type=int, default=800)
    ap.add_argument("--out", default="data/index_ctc.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--seg-max", type=int, default=0,
                    help="split the alignment into segments of at most this many frames, cutting "
                         "at the longest pause inside each window. v9 runs to 123 s per clip; a "
                         "flat duration cap would keep 64%% of its clips but only 27%% of its "
                         "AUDIO, so the long recordings are aligned whole and cut afterwards.")
    ap.add_argument("--seg-min", type=int, default=60,
                    help="discard segments shorter than this many frames (1.5 s)")
    ap.add_argument("--shift", type=int, default=0,
                    help="shift every boundary this many frames later. MEASURE it with "
                         "offset_sweep.py rather than assuming 0: the English corpus peaks at "
                         "+0, but msa-v7 peaks at +1 (8.45 dB vs 7.67 at zero, symmetric about "
                         "+1 over 1,500 clips), i.e. its labels run one 25 ms frame early.")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfgp = "config_student_ar7m.json" if "v9" in (a.manifest or "") or "v7" in (a.manifest or "") \
        else "config_student_wide.json"
    voc = json.load(open(cfgp))["vocab"]
    PAUSE = {voc[c] for c in " ,.;:!?\u060c\u061b\u061f" if c in voc}
    rows = load_rows(a.max_frames, a.manifest)
    if a.limit:
        rows = rows[: a.limit]
    mel = melspec(dev)
    model = Aligner().to(dev).eval()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"])
    print(f"[*] aligning {len(rows):,} clips", flush=True)

    out = open(a.out, "w", encoding="utf-8")
    ok = bad = 0
    scores = []
    for k, r in enumerate(rows):
        try:
            y, _ = sf.read(r["wav"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            n = min(len(y) // HOP, a.max_frames)
            y = y[HOP // 2: HOP // 2 + n * HOP]
            with torch.no_grad():
                m = torch.log(mel(torch.from_numpy(np.ascontiguousarray(y))[None].to(dev)) + 1e-6)
                m = (m - m.mean()) / (m.std() + 1e-5)
                m = m[..., :n]
                lp = F.log_softmax(model(m), dim=-1)
                tgt = torch.tensor([r["ids"]], dtype=torch.long, device=dev)
                lab, sc = AF.forced_align(lp, tgt, blank=BLANK)
            spans = AF.merge_tokens(lab[0], sc[0], blank=BLANK)
            if len(spans) != len(r["ids"]):
                bad += 1; continue
            starts = [min(max(s.start + a.shift, 0), n - 1) for s in spans]
            # token i runs to the start of token i+1: blank frames join the phoneme before them
            ends = starts[1:] + [n]
            dur = [e - s for s, e in zip(starts, ends)]
            lead, tail = starts[0], 0
            ids = [0] + list(r["ids"]) + [0]
            dur = [lead] + dur + [tail]
            keep = [(i, d) for i, d in zip(ids, dur) if d > 0]     # drop empty pad tokens
            ids = [i for i, _ in keep]; dur = [d for _, d in keep]
            if sum(dur) != n or len(ids) < 4:
                bad += 1; continue
            scores.append(float(np.mean([s.score for s in spans])))
            for seg in segment(ids, dur, a.seg_max, a.seg_min, PAUSE):
                out.write(json.dumps({"wav": r["wav"], "start": seg["start"],
                                      "ids": seg["ids"], "dur": seg["dur"],
                                      "text": r["text"], "frames": seg["frames"]},
                                     ensure_ascii=False) + "\n")
                ok += 1
        except Exception:
            bad += 1
        if (k + 1) % 2000 == 0:
            out.flush()
            print(f"  {k+1:,}/{len(rows):,}  ok {ok:,} bad {bad:,}  "
                  f"mean path score {np.mean(scores):.3f}", flush=True)
    out.close()
    print(f"[+] aligned {ok:,}  dropped {bad:,}  mean path score {np.mean(scores):.3f} "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
