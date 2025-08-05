import torch
import torch.nn as nn

from metrics.utils import make_linspace_points, uniform_init_, \
    freeze_rsnorm, get_mean_frequency_2d, simplicity_score


# Notion: c means complexity of function
# ====================== 2. Core function ======================
def compute_simplicity_for_net(
    net: nn.Module,
    input_dim: int,
    scalar_extractor,          # callable: output_tensor(B,...) -> scalar(B)
    grid_range=(-100, 100),
    steps=256,
    amp=1.0,                   # amplitude
    n_samples=100,
    device=None
):
    """
    net      : nn.Module (Actor or Critic)
    scalar_extractor: function that maps model output to 1D tensor [N]
    """
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # 2D grid + zeros for the rest dims
    start = (grid_range[0], grid_range[0])
    end   = (grid_range[1], grid_range[1])
    grid2 = make_linspace_points(start, end, steps, device=device)          # (steps^2, 2)
    if input_dim > 2:
        pad = torch.zeros((grid2.size(0), input_dim-2), device=device)
        full_in = torch.cat([grid2, pad], dim=1)
    else:
        full_in = grid2

    c_list = []

    for _ in range(n_samples):
        # re-init
        net.apply(lambda m: uniform_init_(m, amp, amp))
        freeze_rsnorm(net)

        net.eval()
        with torch.no_grad():
            out = net(full_in)                  # arbitrary shape
            scalar = scalar_extractor(out)      # [steps^2]
            scalar = scalar.view(steps, steps)

        coeff_2d = torch.fft.rfft2(scalar).abs()
        coeff_2d[0, 0] = 0
        coeff_2d = coeff_2d[: coeff_2d.shape[0] // 2 + 1]

        c_val = get_mean_frequency_2d(coeff_2d)
        c_list.append(c_val)

    c_tensor = torch.tensor(c_list, dtype=torch.float32, device=device)
    return simplicity_score(c_tensor), c_tensor