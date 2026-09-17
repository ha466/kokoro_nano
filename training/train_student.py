"""Stage 1: distill Nabra-82M into the 5.25M Kokoro student.

The student is trained on the TEACHER's audio using the TEACHER's own durations, so the
alignment is correct by construction. That is the whole reason for distilling: kitten-arabic
established that with mismatched durations this architecture tops out around WER 0.85 and
62,000 further steps change nothing, because free-running inference beats teacher-forcing --
the durations, not the capacity, were the ceiling.

Adversarial loss is NOT optional. The same project measured reconstruction loss falling from
6.445 to 2.996 while the audio degraded to pure static: multi-resolution STFT and mel distance
constrain magnitude but not phase, so the optimizer can win on the objective and lose the
waveform. MPD + MSD from StyleTTS2 are what make the objective mean something.

Block names mirror KModel (bert/bert_encoder/predictor/text_encoder/decoder) so a finished
student loads straight into KPipeline with the teacher's G2P and voicepack.
"""
from __future__ import annotations
import _env  # noqa: F401
import argparse, json, math, os, random, time
from pathlib import Path

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertConfig

from kokoro.istftnet import Decoder
from kokoro.modules import ProsodyPredictor, TextEncoder, CustomAlbert
from styletts2_gan.discriminators import (MultiPeriodDiscriminator,
                                          MultiResSpecDiscriminator, WavLMDiscriminator)
from styletts2_gan.losses_styletts2 import (GeneratorLoss, DiscriminatorLoss,
                                            MultiResolutionSTFTLoss, WavLMLoss)


class MelL1(nn.Module):
    """L1 on log-mel. The existing multi-resolution STFT loss is spectral convergence on
    LINEAR-frequency bins, so every bin counts equally in Hz; mel spacing instead weights by
    where hearing resolves detail.

    Measured motivation: against the teacher the student runs +0.025 energy in 0-1 kHz and
    -0.025 in 1-4 kHz -- the consonant/F2-F3 band -- with a 12.28 dB mel L1 gap, while
    intelligibility is already fine (WER 0.064, median 0.000). So the deficit is timbre in a
    specific band, which is exactly what a mel-domain term penalises and a linear-bin one
    under-weights."""

    def __init__(self, sr=24000, n_fft=1024, hop=256, n_mels=80):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=1.0)

    def forward(self, y_hat, y):
        a = torch.log(self.mel(y_hat) + 1e-5)
        b = torch.log(self.mel(y) + 1e-5)
        return F.l1_loss(a, b)

SPF = 600            # samples per alignment frame, MEASURED in dump_teacher.py; --sr rewrites it
AUDIO_SR = 0         # set from --audio-sr when the files differ from the model's rate
FRAME_HZ = 40        # the alignment grid, held constant across sample rates so a duration means
                     # the same thing whether the decoder emits 600 or 400 samples per frame
NB = "/home/ahmed-wasfy/Documents/projects/kikiri-tts/training/nabra"


def build_student(cfg_path):
    c = json.load(open(cfg_path, encoding="utf-8"))
    m = nn.Module()
    m.bert = CustomAlbert(AlbertConfig(vocab_size=c["n_token"], **c["plbert"]))
    m.bert_encoder = nn.Linear(m.bert.config.hidden_size, c["hidden_dim"])
    m.predictor = ProsodyPredictor(style_dim=c["style_dim"], d_hid=c["hidden_dim"],
                                   nlayers=c["n_layer"], max_dur=c["max_dur"],
                                   dropout=c["dropout"])
    m.text_encoder = TextEncoder(channels=c["hidden_dim"],
                                 kernel_size=c["text_encoder_kernel_size"],
                                 depth=c["n_layer"], n_symbols=c["n_token"])
    m.decoder = Decoder(dim_in=c["hidden_dim"], style_dim=c["style_dim"],
                        dim_out=c["n_mels"], disable_complex=True, **c["istftnet"])
    return m, c


class DistDataset(Dataset):
    """Teacher audio from the packed i16 memmap, or -- when a row carries "wav" -- Jenny's real
    recording with the DTW-transferred durations from align_real.py.

    For the real branch the clip is cut to sum(dur)*SPF samples, because that is exactly the
    span the durations describe; anything past it has no label."""

    def __init__(self, rows, bin_path, crop):
        self.rows, self.bin_path, self.crop = rows, bin_path, crop
        self._mm = {}

    def _mem(self, path=None):
        # a row may name its own bin file, so a bilingual index can span two teacher dumps
        path = path or self.bin_path
        if path not in self._mm:
            self._mm[path] = np.memmap(path, dtype=np.int16, mode="r")
        return self._mm[path]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        if "wav" in r:
            import soundfile as sf
            # "start" lets one long recording supply many training segments without being cut
            # into files on disk; v9 clips run to 123 s and are segmented at aligner-found pauses.
            need = int(sum(r["dur"])) * SPF
            off = int(r.get("start", 0)) * SPF
            if AUDIO_SR and AUDIO_SR != SPF * FRAME_HZ:
                import soxr
                nspf = AUDIO_SR // FRAME_HZ          # frames are 40 Hz at either rate
                a, _ = sf.read(r["wav"], dtype="float32",
                               start=int(r.get("start", 0)) * nspf,
                               frames=int(sum(r["dur"])) * nspf)
                if a.ndim > 1:
                    a = a.mean(axis=1)
                a = soxr.resample(a, AUDIO_SR, SPF * FRAME_HZ).astype(np.float32)
            else:
                a, _ = sf.read(r["wav"], dtype="float32", start=off, frames=need)
                if a.ndim > 1:
                    a = a.mean(axis=1)
            if len(a) < need:
                a = np.pad(a, (0, need - len(a)))
            a = a[:need]
        else:
            a = np.asarray(self._mem(r.get("bin"))[r["offset"]: r["offset"] + r["samples"]],
                           dtype=np.float32) / 32767.0
        return {"ids": r["ids"], "dur": r["dur"], "audio": a, "lang": r.get("lang", "en")}


def collate(batch):
    B = len(batch)
    L = max(len(b["ids"]) for b in batch)
    ids = torch.zeros(B, L, dtype=torch.long)
    dur = torch.zeros(B, L, dtype=torch.float)
    lens = torch.tensor([len(b["ids"]) for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
        dur[i, : len(b["dur"])] = torch.tensor(b["dur"], dtype=torch.float)
    return {"ids": ids, "dur": dur, "lens": lens,
            "lang": [b.get("lang", "en") for b in batch],
            "audio": [torch.from_numpy(b["audio"]) for b in batch]}


def build_aln(dur, lens, F_max, device):
    """[B,L] integer durations -> [B,L,F] hard monotonic alignment."""
    B, L = dur.shape
    aln = torch.zeros(B, L, F_max, device=device)
    for b in range(B):
        pos = 0
        for j in range(int(lens[b])):
            d = int(dur[b, j])
            if d <= 0 or pos >= F_max:
                continue
            aln[b, j, pos: min(pos + d, F_max)] = 1.0
            pos += d
    return aln


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=40)        # frames; 40 * 600 = 1.0 s
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-disc", type=float, default=2e-4)
    ap.add_argument("--lr-decay", action="store_true",
                    help="cosine-decay the generator LR to 10%% of peak. The 15.34M student "
                         "diverged at a CONSTANT 2e-4: its duration L1 stepped from 0.33 to 0.76 "
                         "between 20.5k and 21k and never recovered, leaving predictions "
                         "uncorrelated with the true durations (r=+0.019 against the 7.48M's "
                         "+0.381). The acoustic path kept improving throughout, so nothing in the "
                         "headline losses showed it.")
    ap.add_argument("--dur-weight", type=float, default=5.0)
    ap.add_argument("--stft-weight", type=float, default=5.0)
    ap.add_argument("--disc-warmup", type=int, default=500)
    ap.add_argument("--slm-weight", type=float, default=0.0,
                    help="WavLM feature-matching loss. Every other term here is MAGNITUDE-domain "
                         "(log-STFT, mel, silence RMS), so nothing but the GAN constrains phase "
                         "-- and our GAN is MPD (waveform) plus MultiResSpec (magnitude, "
                         "phase-blind). Measured: the student matches the teacher on band "
                         "energy, F0 variation, harmonic ratio, modulation spectrum and "
                         "frame-rate buzz, yet sounds robotic. SLM losses are what StyleTTS2 "
                         "credits for naturalness, and this repo vendored them unused.")
    ap.add_argument("--slm-adv-weight", type=float, default=0.0,
                    help="SLM adversarial term on top of feature matching")
    ap.add_argument("--mel-weight", type=float, default=0.0,
                    help="weight on the log-mel L1 term (see MelL1)")
    ap.add_argument("--sil-weight", type=float, default=0.0,
                    help="penalise output energy where the TEACHER is silent. Kokoro-82M emits "
                         "true digital silence in pauses (frame RMS 2e-5); the 5.42M student "
                         "floors at 5.4e-3, 270x higher, which is audible as a constant hiss. "
                         "Multi-resolution STFT is computed on log-magnitude, where -45 dB vs "
                         "-90 dB is a small loss difference and a large perceptual one, so "
                         "nothing in the existing objective punishes it.")
    ap.add_argument("--sil-thresh", type=float, default=1e-3,
                    help="frame RMS below which the teacher counts as silent")
    ap.add_argument("--init-from", default=None,
                    help="load weights from a finished run and start a fresh schedule; --resume "
                         "cannot do this because it restores a scheduler that is already done")
    ap.add_argument("--index", default="dist/index.jsonl")
    ap.add_argument("--bin", default="dist/audio.i16.bin")
    ap.add_argument("--config", default="config_student.json")
    ap.add_argument("--out", default="runs/distill")
    ap.add_argument("--holdout", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--sr", type=int, default=24000,
                    help="output sample rate. 16000 for corpora that are natively 16 kHz: "
                         "upsampling them to 24 kHz invents nothing above 8 kHz and misreports "
                         "the bandwidth. Sets samples-per-frame to sr/40, and the mel, silence "
                         "and WavLM losses with it.")
    ap.add_argument("--audio-sr", type=int, default=0,
                    help="the wav files' own rate, when it differs from --sr. The audio is then "
                         "resampled on read. Durations are in 40 Hz FRAMES, which mean the same "
                         "thing at any rate, so an alignment built at one rate is valid at the "
                         "other and does not need redoing.")
    ap.add_argument("--init-loose", action="store_true",
                    help="load only the tensors whose shapes match. Needed when the decoder is "
                         "rebuilt for another sample rate: the 16 kHz upsample stack differs from "
                         "the 24 kHz one in exactly two tensors, and the other 99.6%% transfers.")
    ap.add_argument("--bilingual", action="store_true",
                    help="rows carry a 'lang' field; condition English rows on af_heart and "
                         "Arabic rows on af_msa so one student learns both, with language "
                         "selectable at inference by choosing the pack")
    ap.add_argument("--two-voices", action="store_true",
                    help="condition on female and male voice packs (voice_female.pt, voice_male.pt)")
    ap.add_argument("--voices-dir", default=None,
                    help="directory containing voice style packs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    global SPF, AUDIO_SR
    SPF = a.sr // FRAME_HZ
    AUDIO_SR = a.audio_sr
    if AUDIO_SR and AUDIO_SR != a.sr:
        print(f"[*] reading {AUDIO_SR} Hz audio, resampling to {a.sr} Hz on the fly", flush=True)
    if a.sr != 24000:
        print(f"[*] {a.sr} Hz: {SPF} samples per frame at {FRAME_HZ} Hz", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(a.index, encoding="utf-8")]
    rows = [r for r in rows if sum(r["dur"]) >= a.crop + 2]
    train_rows, eval_rows = (rows, []) if a.holdout <= 0 else (rows[: -a.holdout], rows[-a.holdout:])
    if not train_rows:
        train_rows = rows
    print(f"[*] train {len(train_rows):,} | held out {len(eval_rows):,}", flush=True)

    model, cfg = build_student(a.config)
    model.to(dev)
    n = sum(p.numel() for p in model.parameters())
    print(f"[*] student {n/1e6:.2f}M", flush=True)

    # Locate default voice pack
    v_dir = Path(a.voices_dir) if a.voices_dir else Path(a.index).parent
    default_pack_path = (v_dir / "af_msa.pt") if (v_dir / "af_msa.pt").exists() \
        else (Path("af_msa.pt") if Path("af_msa.pt").exists() \
        else Path(f"{NB}/af_msa.pt"))
    pack = torch.load(default_pack_path, map_location="cpu", weights_only=True) if default_pack_path.exists() else None
    PACKS = {"ar": pack} if pack is not None else {}
    if a.two_voices:
        f_path = v_dir / "voice_female.pt"
        m_path = v_dir / "voice_male.pt"
        assert f_path.exists() and m_path.exists(), f"Could not find {f_path} and {m_path}"
        PACKS["female"] = torch.load(f_path, map_location="cpu", weights_only=True)
        PACKS["male"] = torch.load(m_path, map_location="cpu", weights_only=True)
        if pack is None:
            pack = PACKS["female"]
        print(f"[*] 2-voice packs loaded from {v_dir}: female ({PACKS['female'].shape}) & male ({PACKS['male'].shape})", flush=True)
    if a.bilingual:
        import _env  # noqa: F401
        from kokoro import KPipeline
        PACKS["en"] = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M",
                                model=False).load_voice("af_heart").cpu()
        if os.path.exists("pack_cs.pt"):
            # the code-switch corpus is a THIRD voice (239.5 Hz vs 292.6 Arabic / 197.5 English),
            # so it gets its own style vector instead of overwriting one of theirs
            PACKS["cs"] = torch.load("pack_cs.pt", map_location="cpu", weights_only=True)
        print(f"[*] bilingual packs: {sorted(PACKS)}", flush=True)
    mpd, msd = MultiPeriodDiscriminator().to(dev), MultiResSpecDiscriminator().to(dev)
    gen_loss = GeneratorLoss(mpd, msd).to(dev)
    disc_loss = DiscriminatorLoss(mpd, msd).to(dev)
    stft_loss = MultiResolutionSTFTLoss().to(dev)
    mel_loss = MelL1(sr=a.sr).to(dev)
    wl = wd = opt_wd = None
    if a.slm_weight > 0 or a.slm_adv_weight > 0:
        # WavLM-base-plus emits 13 hidden states x 768 dims; the discriminator's first conv takes
        # slm_hidden * slm_layers = 9984 channels. Passing (1024, 512, 256) builds a 524288-channel
        # conv and fails on the first real batch.
        wd = WavLMDiscriminator(slm_hidden=768, slm_layers=13, initial_channel=64).to(dev)
        wl = WavLMLoss("microsoft/wavlm-base-plus", wd, a.sr, 16000).to(dev)
        for p_ in wl.wavlm.parameters():
            p_.requires_grad = False
        wl.wavlm.eval()
        opt_wd = torch.optim.AdamW(wd.parameters(), lr=a.lr_disc, betas=(0.8, 0.99))
        print(f"[*] SLM: WavLM feature matching (w={a.slm_weight}) "
              f"adversarial (w={a.slm_adv_weight})", flush=True)

    opt_g = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=1e-2)
    sched_g = (torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=a.steps, eta_min=a.lr * 0.1)
               if a.lr_decay else None)
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()),
                              lr=a.lr_disc, betas=(0.8, 0.99), weight_decay=1e-2)

    dl = DataLoader(DistDataset(train_rows, a.bin, a.crop), batch_size=a.batch, shuffle=True,
                    collate_fn=collate, num_workers=4, drop_last=True, pin_memory=True)

    def run_batch(b, probe=False):
        ids = b["ids"].to(dev); lens = b["lens"].to(dev); durt = b["dur"].to(dev)
        B, L = ids.shape
        tmask = torch.arange(L, device=dev)[None].expand(B, -1) + 1 > lens[:, None]
        langs = b.get("lang", ["female"] * B)
        refs = []
        for l, lg in zip(lens, langs):
            p_ = PACKS.get(lg, pack)
            refs.append(p_[min(int(l) - 3, p_.shape[0] - 1)])
        ref = torch.cat(refs).to(dev)
        s_pred, s_dec = ref[:, 128:], ref[:, :128]

        bert_dur = model.bert(ids, attention_mask=(~tmask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        d = model.predictor.text_encoder(d_en, s_pred, lens, tmask)
        x, _ = model.predictor.lstm(d)
        dur_pred = torch.sigmoid(model.predictor.duration_proj(x)).sum(dim=-1)   # [B,L]
        l_dur = (F.l1_loss(dur_pred, durt, reduction="none") * (~tmask)).sum() / (~tmask).sum()

        frames = durt.sum(dim=1).long()
        Fmax = int(frames.max())
        aln = build_aln(durt, lens, Fmax, dev)
        en = d.transpose(-1, -2) @ aln
        F0, N = model.predictor.F0Ntrain(en, s_pred)
        t_en = model.text_encoder(ids, lens, tmask)
        asr = t_en @ aln
        up = F0.shape[-1] / asr.shape[-1]
        if probe:
            print(f"    asr {tuple(asr.shape)}  F0 {tuple(F0.shape)}  up={up:.2f}  "
                  f"frames={frames.tolist()[:4]}", flush=True)
        assert abs(up - round(up)) < 1e-6, f"F0/asr ratio not integral: {up}"
        up = round(up)

        # Random crop: the decoder is convolutional, so a window is enough and lets the batch
        # be rectangular. Audio, asr and F0/N are cropped on the SAME frame window.
        C = a.crop
        asr_c = torch.empty(B, asr.shape[1], C, device=dev)
        F0_c = torch.empty(B, C * up, device=dev); N_c = torch.empty(B, C * up, device=dev)
        y = torch.empty(B, 1, C * SPF, device=dev)
        for i in range(B):
            hi = max(int(frames[i]) - C, 0)
            f0 = random.randint(0, hi)
            asr_c[i] = asr[i, :, f0:f0 + C]
            F0_c[i] = F0[i, f0 * up:(f0 + C) * up]
            N_c[i] = N[i, f0 * up:(f0 + C) * up]
            w = b["audio"][i]
            seg = w[f0 * SPF:(f0 + C) * SPF]
            if seg.shape[0] < C * SPF:
                seg = F.pad(seg, (0, C * SPF - seg.shape[0]))
            y[i, 0] = seg.to(dev)
        y_hat = model.decoder(asr_c, F0_c, N_c, s_dec)
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(1)
        y_hat = y_hat[..., : y.shape[-1]]
        if probe:
            print(f"    y {tuple(y.shape)}  y_hat {tuple(y_hat.shape)}", flush=True)
        return y, y_hat, l_dur

    if a.probe:
        b = next(iter(dl))
        y, y_hat, l_dur = run_batch(b, probe=True)
        print(f"[+] probe OK  dur_L1 {l_dur.item():.3f}  "
              f"stft {stft_loss(y_hat.squeeze(1), y.squeeze(1)).item():.3f}", flush=True)
        return

    step, best = 0, math.inf
    if a.init_from:
        c = torch.load(a.init_from, map_location="cpu", weights_only=False)
        src = c["model"] if "model" in c else c
        # KModel checkpoints are stored NESTED as {block: {name: tensor}}; the trainer's own
        # last.pt is flat. Flatten so --init-from accepts either, instead of silently matching
        # nothing and training from scratch.
        if src and all(isinstance(v, dict) for v in src.values()):
            src = {f"{b}.{k}": t for b, d in src.items() for k, t in d.items()}
        # Checkpoints saved under DataParallel/DDP carry a "module." prefix inside each block
        # (Nabra-82M does). Left in place it matches nothing and the model trains from scratch.
        if any(".module." in k for k in src):
            src = {k.replace(".module.", ".", 1): v for k, v in src.items()}
        if a.init_loose:
            tgt = model.state_dict()
            fit = {k: v for k, v in src.items() if k in tgt and tgt[k].shape == v.shape}
            skipped = [k for k in tgt if k not in fit]
            model.load_state_dict(fit, strict=False)
            n_fit = sum(tgt[k].numel() for k in fit)
            n_all = sum(v.numel() for v in tgt.values())
            print(f"[*] loose init: {n_fit/1e6:.2f}M of {n_all/1e6:.2f}M "
                  f"({100*n_fit/n_all:.1f}%) loaded, {len(skipped)} tensors left random",
                  flush=True)
            for k in skipped:
                print(f"      random: {k}", flush=True)
        else:
            model.load_state_dict(src)
        if "mpd" in c: mpd.load_state_dict(c["mpd"]); msd.load_state_dict(c["msd"])
        print(f"[*] init weights (+discriminators) from {a.init_from}", flush=True)
    ck = out / "last.pt"
    if a.resume and ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); mpd.load_state_dict(c["mpd"])
        msd.load_state_dict(c["msd"]); opt_g.load_state_dict(c["opt_g"])
        opt_d.load_state_dict(c["opt_d"]); step = c["step"]
        if sched_g is not None:
            if "sched_g" in c:
                sched_g.load_state_dict(c["sched_g"])
            else:                       # checkpoint predates the scheduler
                for _ in range(step):
                    sched_g.step()
        print(f"[*] resumed at step {step}", flush=True)

    t0 = time.time(); acc = {}
    model.train()
    while step < a.steps:
        for b in dl:
            if step >= a.steps:
                break
            y, y_hat, l_dur = run_batch(b)

            opt_d.zero_grad(set_to_none=True)
            d_l = disc_loss(y, y_hat.detach())
            d_l.backward(); torch.nn.utils.clip_grad_norm_(
                list(mpd.parameters()) + list(msd.parameters()), 10.0)
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            l_stft = stft_loss(y_hat.squeeze(1), y.squeeze(1))
            l_mel = (mel_loss(y_hat.squeeze(1), y.squeeze(1)) if a.mel_weight > 0
                     else torch.zeros((), device=dev))
            l_slm = torch.zeros((), device=dev)
            if wl is not None:
                if a.slm_weight > 0:
                    l_slm = wl(y.squeeze(1), y_hat.squeeze(1))
                if a.slm_adv_weight > 0 and step >= a.disc_warmup:
                    l_slm = l_slm + a.slm_adv_weight / max(a.slm_weight, 1e-9) * \
                            wl.generator(y_hat.squeeze(1))
            l_sil = torch.zeros((), device=dev)
            if a.sil_weight > 0:
                H = a.sr // 100          # 10 ms
                n = y.shape[-1] // H
                yr = y[..., : n * H].reshape(y.shape[0], n, H)
                hr = y_hat[..., : n * H].reshape(y_hat.shape[0], n, H)
                yrms = (yr.pow(2).mean(-1) + 1e-12).sqrt()
                tsil = (yrms < a.sil_thresh).float()
                if tsil.sum() > 0:
                    # RMS, not mean-square: the floor is ~5e-3, so squared it is ~3e-5 and
                    # would need a weight of ~10,000 to matter beside an stft term near 1.9.
                    # In RMS the weight reads directly as "how many dB do I care about silence".
                    #
                    # Hinged at the TARGET's own floor. Kokoro emits true digital silence
                    # (1e-6), so against the teacher this is the original loss to 4 decimals.
                    # Jenny is a real microphone: her quiet frames sit at 2.75e-4, and driving
                    # the student below its own target's room tone is not a fix, it is a
                    # different artefact (pumping between speech and dead air).
                    hrms = (hr.pow(2).mean(-1) + 1e-12).sqrt()
                    l_sil = (F.relu(hrms - yrms) * tsil).sum() / tsil.sum()
            # The discriminator is noise until it can tell real from fake; connecting it to the
            # generator immediately injects that noise into the weights.
            l_g = gen_loss(y, y_hat) if step >= a.disc_warmup else torch.zeros((), device=dev)
            loss = a.stft_weight * l_stft + a.dur_weight * l_dur + l_g \
                   + a.sil_weight * l_sil + a.mel_weight * l_mel \
                   + a.slm_weight * l_slm
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_g.step()
            if sched_g is not None:
                sched_g.step()
            step += 1

            for k, v in (("stft", l_stft), ("dur", l_dur), ("gen", l_g), ("disc", d_l),
                         ("sil", l_sil), ("mel", l_mel), ("slm", l_slm)):
                acc[k] = acc.get(k, 0.0) + float(v)
            if step % 100 == 0:
                r = step * a.batch / (time.time() - t0)
                print(f"  step {step}/{a.steps}  " +
                      "  ".join(f"{k} {v/100:.3f}" for k, v in acc.items()) +
                      f"  {r:.0f} clip/s", flush=True)
                acc = {}
            if a.ckpt_every and step % a.ckpt_every == 0:
                tmp = out / "last.pt.tmp"
                torch.save({"model": model.state_dict(), "mpd": mpd.state_dict(),
                            "msd": msd.state_dict(), "opt_g": opt_g.state_dict(),
                            "opt_d": opt_d.state_dict(), "step": step, "config": cfg,
                            **({"sched_g": sched_g.state_dict()} if sched_g else {})}, tmp)
                os.replace(tmp, ck)
                torch.save(model.state_dict(), out / "student.pth")
    # Save a complete self-contained inference bundle in out/
    if a.two_voices:
        import shutil
        data_dir = Path(a.index).parent
        for vf in ("voice_female.pt", "voice_male.pt"):
            src = data_dir / vf
            if src.exists():
                shutil.copy2(src, out / vf)
    if Path(a.config).exists():
        import shutil
        shutil.copy2(a.config, out / "config.json")
    print(f"[+] done at step {step}. Complete model bundle saved to {out}", flush=True)


if __name__ == "__main__":
    main()
