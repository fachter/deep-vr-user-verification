import torch


def mahalanobis_distance(query: torch.tensor, reference: torch.tensor):
    mean = reference.mean(dim=0)
    x_mu = query - mean
    cov_matrix = torch.cov(reference.T)
    inverse_cov = torch.inverse(cov_matrix)
    left_term = x_mu @ inverse_cov
    full_term = left_term @ x_mu.T
    return torch.sqrt(full_term.diagonal() if full_term.dim() == 2 else full_term)
