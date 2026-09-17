"""Synthesize held-out text with the student AND the teacher, free-running.

Free-running is the only honest test: the student predicts its OWN durations, so a duration
predictor that merely memorized total length is exposed here and not by the training loss.
kitten-arabic's ceiling was invisible until exactly this comparison was made.

The teacher is re-scored on the SAME sentences rather than reusing its 0.255 from another
subset, so student and teacher differ only in the model.
"""
from __future__ import annotations
import _env  # noqa: F401
import argparse, json, os, random, sys
import numpy as np, torch, soundfile as sf

from kokoro import KModel, KPipeline

TEACHER = "hexgrad/Kokoro-82M"
SR = 24000


def nest(flat):
    """student.state_dict() (bert.x, decoder.y, ...) -> KModel's {block: state_dict}."""
    out = {}
    for k, v in flat.items():
        blk, rest = k.split(".", 1)
        out.setdefault(blk, {})[rest] = v
    return out


def load_student(weights, config, device):
    ck = f"{weights}.kmodel.pt"
    torch.save(nest(torch.load(weights, map_location="cpu")), ck)
    km = KModel(repo_id="oddadmix/Nabra-82M-v0.1", config=config, model=ck,
                disable_complex=True).eval()
    return km.to(device)


def make_pipe(km):
    return KPipeline(lang_code="a", repo_id=TEACHER, model=km)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/en5m/student.pth")
    ap.add_argument("--config", default="config_student.json")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--tag", default="student")
    ap.add_argument("--out", default="eval")
    ap.add_argument("--with-teacher", action="store_true")
    ap.add_argument("--index", default="dist/index.jsonl",
                    help="held-out split is taken from this index, so pass the SAME "
                         "one the run trained on or the 'held out' clips were seen")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    rows = [json.loads(l) for l in open(a.index, encoding="utf-8")]
    rows = [r for r in rows if sum(r["dur"]) >= 42]
    random.Random(1234).shuffle(rows)                    # same split as train_student.py
    held = rows[-200:][: a.n]

    # The student is conditioned on the pack it was DISTILLED against (af_msa), not on
    # Kokoro's af_heart. Scoring it under af_heart understated it by UTMOS 4.138 -> 3.570
    # and WER 0.0525 -> 0.0701, and made two runs look like regressions that were not.
    pack = torch.load("/home/ahmed-wasfy/Documents/projects/kikiri-tts/training/nabra/af_msa.pt", map_location="cpu", weights_only=True)
    models = {a.tag: load_student(a.weights, a.config, dev)}
    if a.with_teacher:
        models["teacher"] = KModel(repo_id=TEACHER).eval().to(dev)

    meta = []
    for tag, km in models.items():
        pipe = make_pipe(km)
        for i, r in enumerate(held):
            ps, _ = pipe.g2p(r["text"])   # English needs no Arabic normalisation step
            ids = [v for v in (km.vocab.get(p) for p in ps) if v is not None]
            if not ids or len(ids) + 2 > km.context_length:
                continue
            inp = torch.LongTensor([[0, *ids, 0]]).to(dev)
            ref = pack[min(len(ids) - 1, pack.shape[0] - 1)]
            if ref.dim() == 1:
                ref = ref.unsqueeze(0)
            with torch.no_grad():
                audio, dur = km.forward_with_tokens(inp, ref.to(dev), 1.0)
            w = audio.squeeze().float().cpu().numpy()
            sf.write(f"{a.out}/{tag}_{i}.wav", np.clip(w, -1, 1), SR)
            meta.append({"tag": tag, "i": i, "text": r["text"], "sec": len(w) / SR,
                         "teacher_sec": sum(r["dur"]) * 600 / SR})
        print(f"[+] {tag}: {sum(1 for m in meta if m['tag']==tag)} clips", flush=True)
    json.dump(meta, open(f"{a.out}/meta.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
