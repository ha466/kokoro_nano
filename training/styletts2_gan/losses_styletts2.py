"""Losses for StyleTTS 2 / Kokoro student training."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(1024, 2048, 512), hop_sizes=(120, 240, 50), win_lengths=(600, 1200, 240)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x_hat, x):
        # x, x_hat: [B, T]
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            w = torch.hann_window(win, device=x.device)
            spec_x = torch.abs(torch.stft(x, n_fft, hop, win, window=w, return_complex=True))
            spec_x_hat = torch.abs(torch.stft(x_hat, n_fft, hop, win, window=w, return_complex=True))
            # Spectral convergence
            sc = torch.norm(spec_x - spec_x_hat, p="fro") / (torch.norm(spec_x, p="fro") + 1e-7)
            # Log STFT magnitude loss
            mag = F.l1_loss(torch.log(spec_x + 1e-5), torch.log(spec_x_hat + 1e-5))
            total_sc_loss += sc
            total_mag_loss += mag
        return (total_sc_loss + total_mag_loss) / len(self.fft_sizes)


class GeneratorLoss(nn.Module):
    def __init__(self, mpd, msd):
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def forward(self, y, y_hat):
        loss = 0.0
        for disc in (self.mpd, self.msd):
            _, y_d_gs, fmap_rs, fmap_gs = disc(y, y_hat)
            # Adversarial LSGAN loss
            for dg in y_d_gs:
                loss += torch.mean((1.0 - dg) ** 2)
            # Feature matching loss
            for fr, fg in zip(fmap_rs, fmap_gs):
                for f1, f2 in zip(fr, fg):
                    loss += 2.0 * torch.mean(torch.abs(f1.detach() - f2))
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self, mpd, msd):
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def forward(self, y, y_hat):
        loss = 0.0
        for disc in (self.mpd, self.msd):
            y_d_rs, y_d_gs, _, _ = disc(y, y_hat)
            for dr, dg in zip(y_d_rs, y_d_gs):
                r_loss = torch.mean((1.0 - dr) ** 2)
                g_loss = torch.mean(dg ** 2)
                loss += (r_loss + g_loss)
        return loss


class WavLMLoss(nn.Module):
    def __init__(self, model_name, discriminator, sr=24000, target_sr=16000):
        super().__init__()
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.discriminator = discriminator
        self.sr = sr
        self.target_sr = target_sr

    def forward(self, y, y_hat):
        import torchaudio.functional as AF
        if self.sr != self.target_sr:
            y = AF.resample(y, self.sr, self.target_sr)
            y_hat = AF.resample(y_hat, self.sr, self.target_sr)
        with torch.no_grad():
            feat_y = self.wavlm(y, output_hidden_states=True).hidden_states
        feat_y_hat = self.wavlm(y_hat, output_hidden_states=True).hidden_states
        feat_y = torch.cat(feat_y, dim=-1).transpose(1, 2)
        feat_y_hat = torch.cat(feat_y_hat, dim=-1).transpose(1, 2)
        _, fmap_r = self.discriminator(feat_y.detach())
        _, fmap_g = self.discriminator(feat_y_hat)
        loss = 0.0
        for f1, f2 in zip(fmap_r, fmap_g):
            loss += F.l1_loss(f2, f1.detach())
        return loss

    def generator(self, y_hat):
        import torchaudio.functional as AF
        if self.sr != self.target_sr:
            y_hat = AF.resample(y_hat, self.sr, self.target_sr)
        feat = self.wavlm(y_hat, output_hidden_states=True).hidden_states
        feat = torch.cat(feat, dim=-1).transpose(1, 2)
        score, _ = self.discriminator(feat)
        return torch.mean((1.0 - score) ** 2)
