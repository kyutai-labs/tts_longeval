# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
import subprocess as sp

import torch


class Smoother(torch.nn.Module):
    """Smooth some audio to avoid clics.

    Args:
        num_samples: number of samples to smooth on each side.
    """
    def __init__(self, num_samples: int = 256, dtype=None, device=None):
        super().__init__()
        self.num_samples = num_samples
        self.window: torch.Tensor
        self.register_buffer(
            'window',
            torch.hann_window(2 * num_samples, periodic=False, device=device, dtype=dtype),
            persistent=False)

    def forward(self, wav: torch.Tensor, smooth_left: bool = True, smooth_right: bool = True) -> torch.Tensor:
        assert wav.shape[-1] >= 2 * self.num_samples
        left = wav[..., :self.num_samples]
        if smooth_left:
            left = left * self.window[:self.num_samples]
        right = wav[..., -self.num_samples:]
        if smooth_right:
            right = right * self.window[self.num_samples:]

        middle = wav[..., self.num_samples:-self.num_samples]
        return torch.cat([left, middle, right], dim=-1)


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format.
    """
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def _piping_to_ffmpeg(out_path: Path | str, wav: torch.Tensor, sample_rate: int, flags: list[str]):
    # ffmpeg is always installed and torchaudio is a bit unstable lately, so let's bypass it entirely.
    assert wav.dim() == 2, wav.shape
    command = [
        'ffmpeg',
        '-loglevel', 'error',
        '-y', '-f', 'f32le', '-ar', str(sample_rate), '-ac', str(wav.shape[0]),
        '-i', '-'] + flags + [str(out_path)]
    input_ = f32_pcm(wav).t().detach().cpu().numpy().tobytes()
    sp.run(command, input=input_, check=True)


def audio_write(filename: str | Path,
                wav: torch.Tensor, sample_rate: int,
                mp3_rate: int = 320, ogg_rate: int | None = None) -> Path:
    """Convenience function for saving audio to disk. Returns the filename the audio was written to.

    Args:
        filename (str or Path): Filename with the extension.
        wav (torch.Tensor): Audio data to save.
        sample_rate (int): Sample rate of audio data.
        format (str): Either "wav", "mp3", "ogg", or "flac".
        mp3_rate (int): kbps when using mp3s.
        ogg_rate (int): kbps when using ogg/vorbis. If not provided, let ffmpeg decide for itself.
    Returns:
        Path: Path of the saved audio.
    """
    assert wav.dtype.is_floating_point, "wav is not floating point"
    filename = Path(filename)
    assert isinstance(filename, Path)
    wav = wav.float()
    if wav.dim() == 1:
        wav = wav[None]
    elif wav.dim() > 2:
        raise ValueError("Input wav should be at most 2 dimension.")
    assert wav.isfinite().all()
    wav = wav.clamp(-0.99, 0.99)
    suffix = filename.suffix
    if suffix == '.mp3':
        flags = ['-f', 'mp3', '-c:a', 'libmp3lame', '-b:a', f'{mp3_rate}k']
    elif suffix == '.wav':
        suffix = '.wav'
        flags = ['-f', 'wav', '-c:a', 'pcm_s16le']
    elif suffix == '.flac':
        flags = ['-c:a', 'flac']
    elif suffix == '.ogg':
        flags = ['-f', 'ogg', '-c:a', 'libvorbis']
        if ogg_rate is None:
            if sample_rate <= 24000:
                ogg_rate = 64
            else:
                ogg_rate = 128
            flags += ['-b:a', f'{ogg_rate}k']
    else:
        raise RuntimeError(f"Invalid suffix {suffix}. Only wav or mp3 are supported.")
    try:
        _piping_to_ffmpeg(filename, wav, sample_rate, flags)
    except Exception:
        if filename.exists():
            # we do not want to leave half written files around.
            filename.unlink()
        raise
    return filename
