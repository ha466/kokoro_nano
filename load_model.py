"""Load Kokoro-7M-Distill, a 7.48M English student distilled from hexgrad/Kokoro-82M.

The patched kokoro package is vendored here because the config sets decoder
`hidden_channels`/`out_channels`, which upstream hardcodes at 1024/512 -- stock `kokoro` raises
TypeError on this config. Defaults in the patch reproduce the stock model, so the 82M teacher
still loads through it unchanged.

    from load_model import load
    model, pipeline, voice = load()
    audio = next(pipeline("Hello, this is a small English voice.", voice=voice))[2]

The default voice is `af_msa.pt` because that is the style pack the student was CONDITIONED on
during distillation. Earlier revisions defaulted to `af_heart.pt` -- Kokoro's own pack, which
this model never saw. Measured on 60 held-out sentences, that mismatch cost UTMOS 4.138 -> 3.570
and WER 0.0525 -> 0.0701. `af_heart.pt` is kept only so older code does not break.
"""
import os, sys, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "kokoro_patched"))    # must precede any kokoro import

from kokoro import KModel, KPipeline                        # noqa: E402

REPO = "oddadmix/Kokoro-7M-Distill"


def load(device=None, weights="kokoro_en_7m.pth", config="config.json", voice="af_msa.pt"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = KModel(repo_id=REPO, config=os.path.join(HERE, config),
               model=os.path.join(HERE, weights), disable_complex=True).eval().to(device)
    p = KPipeline(lang_code="a", repo_id=REPO, model=m)
    return m, p, torch.load(os.path.join(HERE, voice), map_location="cpu", weights_only=True)
