"""Offset sweep + reversed-label control. HOP must match the corpus's sample rate:
600 at 24 kHz, 400 at 16 kHz. Reading a 24 kHz corpus on the 16 kHz grid shows no
alignment at all, which looks exactly like a failed gate."""
import json, sys, numpy as np, soundfile as sf
HOP = int(sys.argv[3]) if len(sys.argv) > 3 else 400
cfg = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "config_student_16k.json"))
inv = {v: k for k, v in cfg["vocab"].items()}
V = set("aeiouɑɐɒæɔəɚɛɜɪʊʌɤøœɨɯ"); Q = set(" ,.;:!?،؛؟")
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")][:900]


def gap(shift=0, rev=False):
    a, b = [], []
    for r in rows:
        n = r["frames"]
        y, _ = sf.read(r["wav"], dtype="float32", start=int(r.get("start", 0)) * HOP,
                       frames=n * HOP)
        if y.ndim > 1: y = y.mean(1)
        if len(y) < n * HOP: y = np.pad(y, (0, n * HOP - len(y)))
        e = 20 * np.log10(np.sqrt((y.reshape(n, HOP) ** 2).mean(1) + 1e-12))
        ids = list(r["ids"])[::-1] if rev else r["ids"]
        pos = 0
        for pid, d in zip(ids, r["dur"]):
            ch = inv.get(pid, "")
            seg = e[max(pos + shift, 0): max(pos + d + shift, 0)]
            if len(seg):
                if ch in V: a.append(seg.mean())
                elif ch in Q: b.append(seg.mean())
            pos += d
    return (np.mean(a) - np.mean(b)) if a and b else float("nan")


gs = {s: gap(s) for s in (-3, -2, -1, 0, 1, 2, 3)}
best = max(gs, key=lambda k: gs[k])
print("sweep " + "  ".join(f"{s:+d}:{g:5.2f}" for s, g in gs.items()) + f"   peak at {best:+d}")
print(f"TRUE labels      gap {gs[0]:+6.2f} dB")
print(f"REVERSED control gap {gap(0, True):+6.2f} dB")
