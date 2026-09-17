"""A CTC phoneme aligner trained on Jenny's OWN audio, to replace the DTW duration transfer.

Why this exists. DTW-transferring the teacher's durations onto Jenny passed the energy gate
(+4.34 dB vowel-minus-quiet against a -0.18 dB reversed-label control) and still produced bad
labels: finetuning on them tripled free-running WER (0.041 -> 0.145), and -- the giveaway --
teacher-forcing those same durations was WORSE than free-running (0.258), which can only happen
when the labels disagree with the audio. Feeding the DTW durations to the ALREADY-GOOD shipped
model degraded it too (0.041 -> 0.079), so the durations, not the student, are the fault.

The energy gate is necessary but not sufficient: it only asks whether loud frames land on
vowels, which survives several frames of per-phoneme slop.

CTC needs only the phoneme SEQUENCE, not its timing, so this trains directly on Jenny's real
recordings -- no synthetic-to-real domain gap, unlike an aligner trained on teacher audio.
Framing is locked to the student's 40 Hz grid (hop 600 at 24 kHz) so a frame index means the
same thing here and in the trainer; converting rates between the two is where a previous
project in this line lost a run.
"""
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

SR, HOP, NMELS = 24000, 600, 80
NVOCAB = 178          # Kokoro token ids 0..177
BLANK = NVOCAB        # CTC blank gets its own class


def melspec(dev):
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1200, hop_length=HOP, n_mels=NMELS, center=True, power=2.0).to(dev)


class Aligner(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(NMELS, d, 5, padding=2), nn.GELU(), nn.BatchNorm1d(d),
            nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.BatchNorm1d(d))
        self.rnn = nn.LSTM(d, d, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * d, NVOCAB + 1)

    def forward(self, mel):                    # mel [B, NMELS, T]
        x = self.conv(mel).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.out(x)                     # [B, T, NVOCAB+1]


class ClipSet(Dataset):
    """Jenny audio paired with the teacher's G2P phoneme ids for the same sentence."""

    def __init__(self, rows, max_frames):
        self.rows, self.max_frames = rows, max_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        y, _ = sf.read(r["wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        n = min(len(y) // HOP, self.max_frames)
        # frame j must cover samples [j*HOP, (j+1)*HOP) to match the student's grid; a
        # centered STFT centres frame j at j*HOP, so drop half a hop first.
        y = y[HOP // 2: HOP // 2 + n * HOP]
        return {"y": torch.from_numpy(np.ascontiguousarray(y)), "ids": r["ids"], "n": n}


def collate(b):
    T = max(x["n"] for x in b)
    L = max(len(x["ids"]) for x in b)
    y = torch.zeros(len(b), T * HOP)
    tgt = torch.zeros(len(b), L, dtype=torch.long)
    ylen = torch.tensor([x["n"] for x in b], dtype=torch.long)
    tlen = torch.tensor([len(x["ids"]) for x in b], dtype=torch.long)
    for i, x in enumerate(b):
        y[i, : len(x["y"])] = x["y"]
        tgt[i, : len(x["ids"])] = torch.tensor(x["ids"], dtype=torch.long)
    return y, tgt, ylen, tlen


def load_rows(max_frames, manifest=None):
    """Rows of {wav, ids, text, n}.

    A manifest that already carries phoneme ids (build_v7.py writes these) is used directly.
    Otherwise fall back to joining Jenny's audio against the teacher's G2P by text, which is how
    the English corpus was assembled."""
    rows = []
    if manifest:
        for l in open(manifest, encoding="utf-8"):
            d = json.loads(l)
            p = [i for i in d["ids"] if i != 0]
            n = d.get("frames") or int(d["duration"] * SR) // HOP
            # DROP clips past the cap, never truncate them: the phoneme sequence describes the
            # whole recording, so a clipped waveform paired with the full label set teaches the
            # aligner to compress. v9 is long-form (up to 123 s), where this actually bites.
            if not p or n > max_frames or n < max(len(p) + 2, 42):
                continue
            rows.append({"wav": d["wav"], "ids": p, "text": d["text"], "n": n})
        return rows
    ids = {}
    for l in open("dist/index.jsonl", encoding="utf-8"):
        d = json.loads(l)
        ids[d["text"]] = [i for i in d["ids"] if i != 0]       # strip the $ pad tokens
    for l in open("data/jenny.jsonl", encoding="utf-8"):
        d = json.loads(l)
        t = d["text"].strip().replace("\n", " ")
        p = ids.get(t)
        if not p:
            continue
        n = min(int(d["duration"] * SR) // HOP, max_frames)
        if n < len(p) + 2 or n < 42:            # CTC needs at least one frame per token
            continue
        rows.append({"wav": d["audio"], "ids": p, "text": t, "n": n})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-frames", type=int, default=800)      # 20 s
    ap.add_argument("--manifest", default=None,
                    help="jsonl with wav/ids/frames already resolved")
    ap.add_argument("--out", default="runs/aligner")
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(a.max_frames, a.manifest)
    random.Random(1234).shuffle(rows)
    tr, ev = rows[: -a.holdout], rows[-a.holdout:]
    print(f"[*] aligner train {len(tr):,} | held out {len(ev):,}", flush=True)

    mel = melspec(dev)
    model = Aligner().to(dev)
    print(f"[*] aligner {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                                pct_start=0.05)
    step = 0
    ck = out / "last.pt"
    if a.resume and ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"]); step = c["step"]
        print(f"[*] resumed at {step}", flush=True)

    dl = DataLoader(ClipSet(tr, a.max_frames), batch_size=a.batch, shuffle=True,
                    collate_fn=collate, num_workers=4, drop_last=True, pin_memory=True)
    t0 = time.time(); acc = 0.0; nacc = 0
    model.train()
    while step < a.steps:
        for y, tgt, ylen, tlen in dl:
            if step >= a.steps:
                break
            y, tgt = y.to(dev, non_blocking=True), tgt.to(dev)
            with torch.no_grad():
                m = torch.log(mel(y) + 1e-6)
                m = (m - m.mean(dim=(1, 2), keepdim=True)) / (m.std(dim=(1, 2), keepdim=True) + 1e-5)
                m = m[..., : int(ylen.max())]
            lp = F.log_softmax(model(m), dim=-1).transpose(0, 1)      # [T,B,C]
            loss = F.ctc_loss(lp, tgt, ylen.to(dev), tlen.to(dev), blank=BLANK,
                              zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); step += 1
            acc += float(loss); nacc += 1
            if step % 200 == 0:
                print(f"  step {step}/{a.steps}  ctc {acc/nacc:.4f}  "
                      f"{step*a.batch/(time.time()-t0):.0f} clip/s", flush=True)
                acc = 0.0; nacc = 0
            if step % 2000 == 0 or step == a.steps:
                tmp = out / "last.pt.tmp"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step}, tmp)
                os.replace(tmp, ck)
    print(f"[+] aligner done at {step}", flush=True)


if __name__ == "__main__":
    main()
