"""Re-split index segments that exceed plbert's 512 position embeddings.

The frame cap alone is not sufficient: a 20 s segment of dense Arabic can carry 669 phonemes,
and ALBERT's positional table stops at 512. Splitting at a pause inside the segment keeps the
audio (148 segments, 0.3% of the corpus) instead of dropping it, and keeps every cut on a
silence boundary the aligner found.
"""
import json, sys
from align_ctc import segment            # same pause-aware splitter used during alignment

CFG = json.load(open("config_student_ar7m.json"))
PAUSE = {CFG["vocab"][c] for c in " ,.;:!?،؛؟" if c in CFG["vocab"]}
MAX_TOK = 480                            # headroom under 512 for the two boundary tokens
src, dst = sys.argv[1], sys.argv[2]
out = open(dst, "w", encoding="utf-8")
kept = split = dropped = 0
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    if len(r["ids"]) <= MAX_TOK:
        out.write(line); kept += 1; continue
    # split by frames proportionally until every piece is under the token cap
    target = max(60, int(r["frames"] * MAX_TOK / len(r["ids"]) * 0.9))
    base = int(r.get("start", 0))
    for s in segment(r["ids"], r["dur"], target, 60, PAUSE):
        if len(s["ids"]) > MAX_TOK or s["frames"] < 42:
            dropped += 1; continue
        out.write(json.dumps({"wav": r["wav"], "start": base + s["start"], "ids": s["ids"],
                              "dur": s["dur"], "text": r["text"], "frames": s["frames"]},
                             ensure_ascii=False) + "\n")
        split += 1
out.close()
print(f"[+] kept {kept:,}  re-split into {split:,}  dropped {dropped:,} -> {dst}", flush=True)
