import torch
from torch.nn import functional as F


def pad_col(input, val=0, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")


def nll_pmf(phi: torch.tensor, idx_durations: torch.tensor, events: torch.tensor,
            epsilon: float = 1e-7) -> torch.Tensor:
    """Negative log-likelihood of the discrete time hazard parametrized model LogisticHazard [1].

    :param phi: Estimates in (-inf, inf), where hazard = sigmoid(phi). Probability of an even occurring within a given time bin. batch_size x num_bins
    :type phi: torch.Tensor
    :param idx_durations: Event times represented as indices. Defines whether an even occurred within the time bin. batch_size x num_bins
    :type idx_durations: torch.Tensor
    :param events: Indicator of event (1.) or censoring (0.). batch_size x 1
    :type events: torch.Tensor
    :return: cox hazard loss
    :rtype: torch.Tensor
    """
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`." +
                         f" Need at least `phi.shape[1] = {idx_durations.max().item() + 1}`," +
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    # idx_durations = idx_durations.view(-1, 1)
    phi = pad_col(phi)
    gamma = phi.max(1)[0]
    cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
    sum_ = cumsum[:, -1]
    part1 = phi.gather(1, idx_durations).sub(gamma).mul(events)
    part2 = - sum_.relu().add(epsilon).log()
    part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
    # need relu() in part3 (and possibly part2) because cumsum on gpu has some bugs and we risk getting negative numbers.
    loss = - part1.add(part2).add(part3)
    return loss.mean()
