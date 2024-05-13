import torch
from torch.distributions import MultivariateNormal
from pytorch_metric_learning.distances import BaseDistance
from typing import Union, Tuple

SliceType = Union[
    Tuple[None, None, slice],
    Tuple[slice, None],
    Tuple[None, slice],
    slice,
    None,
]


def get_cov_matrix_from_variance_as_cholesky(variance: torch.tensor) -> torch.tensor:
    dimensions = variance.size(-1)
    covariance_matrix = variance[:, None, :] * torch.eye(dimensions, device=variance.device).reshape((1, dimensions, dimensions))
    try:
        return torch.linalg.cholesky(covariance_matrix)
    except Exception as e:
        raise e


def get_distributions_from_embeddings(query_emb, ref_emb,
                                      query_slice: SliceType = slice(None),
                                      ref_slice: SliceType = slice(None)):
    cut = query_emb.size(-1) // 2
    query_mean = query_emb[..., :cut]
    query_var = query_emb[..., cut:]
    ref_mean = ref_emb[..., :cut]
    ref_var = ref_emb[..., cut:]

    query_distributions = MultivariateNormal(
        loc=query_mean[query_slice],
        scale_tril=get_cov_matrix_from_variance_as_cholesky(query_var)[query_slice]
    )
    ref_distributions = MultivariateNormal(
        loc=ref_mean[ref_slice],
        scale_tril=get_cov_matrix_from_variance_as_cholesky(ref_var)[ref_slice]
    )

    return query_distributions, ref_distributions


class KLDivDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb: torch.tensor, ref_emb: torch.tensor):
        query_dim = query_emb.size(-1)
        ref_dim = ref_emb.size(-1)
        assert query_dim == ref_dim, \
            f"Query and Reference embeddings need to be of same dimension ({query_dim} != {ref_dim})"
        query_slice = slice(None), None
        ref_slice = None
        query_distributions, ref_distributions = get_distributions_from_embeddings(query_emb, ref_emb, query_slice,
                                                                                   ref_slice)

        return torch.distributions.kl.kl_divergence(query_distributions, ref_distributions)

    def pairwise_distance(self, query_emb: torch.tensor, ref_emb: torch.tensor):
        query_dim = query_emb.size(-1)
        ref_dim = ref_emb.size(-1)
        assert query_dim == ref_dim, \
            f"Query and Reference embeddings need to be of same dimension ({query_dim} != {ref_dim})"
        query_length = query_emb.size(0)
        ref_length = ref_emb.size(0)
        assert query_length == ref_length or query_length == 1 or ref_length == 1, \
            (f"Query and Reference length have to "
             f"be equal or one of them has to be of length 1 ({query_length} != {ref_length})")

        query_distributions, ref_distributions = get_distributions_from_embeddings(query_emb, ref_emb)
        return torch.distributions.kl.kl_divergence(query_distributions, ref_distributions)
