import torch
import torch.nn as nn


# ====================== 1. Utils ======================
def make_linspace_points(start, end, steps, device=None, dtype=torch.float32):
    axes = [torch.linspace(s, e, steps, device=device, dtype=dtype)
            for s, e in zip(start, end)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    pts = torch.stack([m.reshape(-1) for m in mesh], dim=1)
    return pts  # (steps**D, D)

# TODO: apply Nyquist-Shannon limit to sample k
def get_mean_frequency_2d(coeff_2d: torch.Tensor):
    """Get the mean frequency of the 2D Fourier coefficients."""

    dim_u, dim_v = coeff_2d.shape
    max_freq = round(((dim_u - 1) ** 2 + (dim_v - 1) ** 2) ** (1 / 2))

    grid_i, grid_j = torch.meshgrid(
        torch.Tensor(range(dim_u)), torch.Tensor(range(dim_v)), indexing="ij"
    )
    bins = torch.round(torch.sqrt(grid_i**2 + grid_j**2)).to(torch.int)

    coeff_1d = torch.zeros(max_freq + 1)

    for i in range(max_freq + 1):
        coeff_1d[i] = coeff_2d[bins == i].sum()

    # Normalize
    coeff_1d[0] = 0  # ignore shift coming from 0 "Hz" frequency.
    coeff_1d /= coeff_1d.sum()

    # Compute mean frequency
    mean_freq = (coeff_1d * torch.Tensor(range(max_freq + 1))).sum() / max_freq

    return mean_freq.item()

def simplicity_score(c_vals: torch.Tensor):
    return (1.0 / c_vals).mean().item()

def uniform_init_(module, w_amp, b_amp=None):
    if b_amp is None: b_amp = w_amp
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -w_amp, w_amp)
        if module.bias is not None:
            nn.init.uniform_(module.bias, -b_amp, b_amp)

def freeze_rsnorm(model):
    from types import MethodType
    for m in model.modules():
        if m.__class__.__name__ == "RSNorm":   # 직접 isinstance(RSNorm) 해도 됨
            def forward_no_update(self, x):
                squeezed = 0
                if x.dim() == 1:
                    x = x.unsqueeze(0); squeezed = 1
                y = (x - self.mu) / torch.sqrt(self.var + self.eps)
                return y.squeeze(0) if squeezed else y
            m.forward = MethodType(forward_no_update, m)