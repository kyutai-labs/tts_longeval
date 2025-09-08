# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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
