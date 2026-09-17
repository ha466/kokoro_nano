"""Standard StyleTTS 2 / HiFi-GAN discriminators for adversarial TTS training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorP(nn.Module):
    """Single period sub-discriminator."""
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD)."""
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    """Single spectrogram sub-discriminator."""
    def __init__(self, n_fft=1024, hop_length=120, win_length=600):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def spect(self, x):
        w = torch.hann_window(self.win_length, device=x.device)
        spec = torch.stft(x.squeeze(1), self.n_fft, self.hop_length, self.win_length,
                          window=w, return_complex=True)
        return torch.abs(spec).unsqueeze(1)

    def forward(self, x):
        fmap = []
        x = self.spect(x)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiResSpecDiscriminator(nn.Module):
    """Multi-Resolution Spectrogram Discriminator (MSD)."""
    def __init__(self, configs=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(n_fft, hop, win) for n_fft, hop, win in configs
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WavLMDiscriminator(nn.Module):
    """WavLM representation discriminator for natural phase & prosody."""
    def __init__(self, slm_hidden=768, slm_layers=13, initial_channel=64):
        super().__init__()
        in_ch = slm_hidden * slm_layers
        self.conv = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, initial_channel, 5, padding=2)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(initial_channel, initial_channel * 2, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(initial_channel * 2, initial_channel * 4, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(initial_channel * 4, 1, 3, padding=1))
        )

    def forward(self, x):
        fmap = []
        for l in self.conv:
            x = l(x)
            fmap.append(x)
        return torch.flatten(x, 1, -1), fmap
