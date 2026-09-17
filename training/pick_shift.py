"""Measure the frame offset of an alignment and print the correction to apply.

Run per corpus, never assumed: English peaked at +0, msa-v7 at +1 under the same code. An
uncorrected one-frame offset is a milder version of the DTW failure that tripled WER.
"""
import json, sys
import numpy as np, soundfile as sf
HOP = 600
cfg = json.load(open(sys.argv[2]))
inv = {v: k for k, v in cfg["vocab"].items()}
VOWELS = set("aeiouɑɐɒæɔəɚɛɜɪʊʌɤøœɨɯ")
QUIET = set(" ,.;:!?،؛؟")
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")][:900]


def gap(shift):
    v, q = [], []
    for r in rows:
        n = r["frames"]
        y, _ = sf.read(r["wav"], dtype="float32", start=int(r.get("start", 0)) * HOP,
                       frames=n * HOP)
        if y.ndim > 1: y = y.mean(1)
        if len(y) < n * HOP: y = np.pad(y, (0, n * HOP - len(y)))
        e = 20 * np.log10(np.sqrt((y.reshape(n, HOP) ** 2).mean(1) + 1e-12))
        pos = 0
        for pid, d in zip(r["ids"], r["dur"]):
            ch = inv.get(pid, "")
            seg = e[max(pos + shift, 0): max(pos + d + shift, 0)]
            if len(seg):
                if ch in VOWELS: v.append(seg.mean())
                elif ch in QUIET: q.append(seg.mean())
            pos += d
    return (np.mean(v) - np.mean(q)) if v and q else float("nan")


gs = {s: gap(s) for s in (-3, -2, -1, 0, 1, 2, 3)}
best = max(gs, key=lambda k: gs[k])
print("sweep " + "  ".join(f"{s:+d}:{g:5.2f}" for s, g in gs.items()), file=sys.stderr)
print(best)
