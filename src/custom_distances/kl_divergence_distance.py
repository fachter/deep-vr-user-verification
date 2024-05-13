from pytorch_metric_learning.distances import BaseDistance
from torch.distributions import MultivariateNormal, kl_divergence
import torch


class KLDivDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert not self.is_inverted

    def compute_mat(self, query_emb: torch.Tensor, ref_emb):
        return kl_divergence_variant_2(query_emb[:, None], ref_emb[None])

    def pairwise_distance(self, query_emb, ref_emb):
        return kl_divergence_variant_2(query_emb, ref_emb)


def kl_divergence_variant_2(x1: torch.Tensor, x2: torch.Tensor):
    k = x1.size(-1) // 2
    mean_x1 = x1[..., :k]
    mean_x2 = x2[..., :k]

    var_x1 = x1[..., k:]
    var_x2 = x2[..., k:]

    term_1 = (var_x2 ** -1 * var_x1).sum(axis=-1)
    mean_difference = (mean_x2 - mean_x1)
    term_2 = (mean_difference * var_x2 ** -1 * mean_difference).sum(axis=-1)
    term_3 = var_x2.log().sum(axis=-1) - var_x1.log().sum(axis=-1)

    final = 0.5 * (term_1 + term_2 - k + term_3)

    return final


def _kl_normal_normal(p_mean, p_variance, q_mean, q_variance):
    k = p_mean.size(-1)
    var_ratio = (p_variance / q_variance).pow(2)
    t1 = ((p_mean - q_mean) / q_variance).pow(2)
    return 0.5 * (var_ratio + t1 - k - var_ratio.log())


def kl_divergence_one_by_two_matrix_embeddings(query_embedding: torch.Tensor,
                                               ref_embedding: torch.Tensor) -> torch.Tensor:
    cut = int(query_embedding.size(-1) / 2)
    m1 = query_embedding[:cut]
    v1 = torch.eye(cut) * query_embedding[cut:]
    mn1 = MultivariateNormal(m1, v1)
    total_div = 0
    for reference in ref_embedding:
        m2 = reference[:cut]
        v2 = torch.eye(cut) * reference[cut:]
        mn2 = MultivariateNormal(m2, v2)
        div = kl_divergence(mn1, mn2)
        total_div = total_div + div
    total_div = total_div / len(ref_embedding)
    return total_div


def kl_divergence_two_matrix_embeddings(query_embedding: torch.Tensor, ref_embedding: torch.Tensor) -> torch.Tensor:
    total_div = 0
    for query, reference in zip(query_embedding, ref_embedding):
        div = kl_divergence_single_embeddings(query, reference)
        total_div = total_div + div
    total_div = total_div / len(query_embedding)
    return total_div


def kl_divergence_single_embeddings(query, reference):
    cut = int(query.size(-1) / 2)
    m1 = query[:cut]
    eye1 = torch.eye(cut)
    eye2 = torch.eye(cut)
    if query.get_device() != -1:
        eye1 = eye1.to(device=query.get_device())
        eye2 = eye2.to(device=query.get_device())
    v1 = eye1 * query[cut:]
    m2 = reference[:cut]
    v2 = eye2 * reference[cut:]
    mn1 = MultivariateNormal(m1, v1)
    mn2 = MultivariateNormal(m2, v2)
    div = kl_divergence(mn1, mn2)
    return div


if __name__ == '__main__':
    n_dimensions = 4
    n_embeddings = 5
    new_query = torch.rand((n_dimensions,), requires_grad=True)
    new_ref = torch.rand((n_embeddings, n_dimensions), requires_grad=True)
    results_vector_with_matrix = kl_divergence_variant_2(new_query, new_ref)
    print("Vector * Matrix", results_vector_with_matrix.size())

    full_query = torch.rand((n_embeddings, n_dimensions), requires_grad=True)
    full_result = kl_divergence_variant_2(full_query, new_ref)
    print("Full", full_result.size())

    full_result.mean().backward()
    print("Full working")
    results_vector_with_matrix.mean().backward()
    print("Vector working")

    dist_func = KLDivDistance()
    dists = dist_func(
        torch.tensor([
            [-0.1, 0.1, 1.1, 1.1],
            [58., 12., 1., 0.9],
            [0., 0., 5., 2],
            [0., 0., 0.2, 0.5],
            [3., -5., 1.5, 0.5]
        ]),
        torch.tensor([[0., 0., 1., 1.]]))
    print(dists)
    print(new_ref)
