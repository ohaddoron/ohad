import torch

reduction_fn = dict(
    mean=torch.mean,
    sum=torch.sum
)


def contrastive_loss(input_1: torch.Tensor, input_2: torch.Tensor, reduction: str = 'mean', tau: torch.Tensor = 1.):
    """
    https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
    """
    similarity_mat = sim_matrix(input_1, input_2)

    n = len(similarity_mat)

    nominator = torch.exp(similarity_mat / tau)
    denominator = torch.sum(
        torch.exp((similarity_mat.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)) / tau))
    return reduction_fn.get(reduction)(-torch.log(nominator / denominator))


def sim_matrix(a: torch.Tensor, b: torch.Tensor, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
