import torch
from pytorch_metric_learning.distances import BaseDistance, LpDistance, CosineSimilarity

from src.custom_distances import KLDivDistance
from src.verification_heads.distance_match_probability_verification_head import DistanceMatchProbabilityVerificationHead
import numpy as np


class SamplingMatchProbabilityVerificationHead(DistanceMatchProbabilityVerificationHead):
    def __init__(self, distance: BaseDistance, k=8, threshold=None):
        super().__init__(distance, threshold=threshold)
        self._k = k
        assert isinstance(self.distance, LpDistance) or isinstance(self.distance, CosineSimilarity), \
            "The comparison after sampling the distributions only works with LpDistance or CosineSimilarity"

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, *_args):
        query_distributions = _monte_carlo_sampling(query_embeddings, self._k)
        reference_distributions = _monte_carlo_sampling(reference_embeddings, self._k)
        distances = self._distance_forward_in_batches(query_distributions, reference_distributions)
        matches = self._match_probability(distances).mean(dim=-1)
        return matches

def _monte_carlo_sampling(embeddings: torch.tensor, k: int):
    cut = embeddings.size(-1) // 2
    mean = embeddings[:, :cut]
    sigma = embeddings[:, cut:]
    epsilon = torch.randn((k,) + mean.size(), device=mean.device).unsqueeze(-2)
    sigma_matrix = torch.diag_embed(sigma)
    # distributions = []
    all_samples = []
    for step_index in range(int(np.ceil(embeddings.size(0) / 1000.))):
        samples = (mean[step_index * 1000:(step_index + 1) * 1000].unsqueeze(1)
                   + epsilon[:, step_index * 1000:(step_index + 1) * 1000]
                   @ sigma_matrix[step_index * 1000:(step_index + 1) * 1000]).squeeze(2)
        all_samples.append(samples.permute(1, 0, 2))
        # sample_means.append(samples.mean(dim=0))
        # sample_vars.append(samples.var(dim=0))

    # samples = (mean.unsqueeze(1) + epsilon @ sigma_matrix).squeeze(2)
    # distributions = torch.concat([
    #     torch.concat(sample_means, dim=0),
    #     torch.concat(sample_vars, dim=0)
    # ], dim=-1)
    return torch.concat(all_samples, dim=0)


def _hib_loss(scorer, distributions, labels):
    same_probs = scorer(distributions, distributions)  # (B, B).
    same_mask = labels[None] == labels[:, None]  # (B, B).
    positive_probs = same_probs[same_mask]
    negative_probs = same_probs[~same_mask]
    positive_xent = torch.nn.functional.binary_cross_entropy(positive_probs, torch.ones_like(positive_probs))
    negative_xent = torch.nn.functional.binary_cross_entropy(negative_probs, torch.zeros_like(negative_probs))
    return 0.5 * (positive_xent + negative_xent)


def main():
    dimensions = 4
    device = torch.device("cuda")
    query_targets = torch.randint(0, 5, (3000,), device=device)
    ref_targets = torch.randint(0, 5, (5000,), device=device)
    x_query = torch.rand((3_000, dimensions), device=device)
    x_ref = torch.rand((5_000, dimensions), device=device)
    targets = query_targets[None].eq(ref_targets[:, None])

    head = SamplingMatchProbabilityVerificationHead(LpDistance(), 8).to(device)
    loss = _hib_loss(head, x_query, query_targets)
    loss.backward()

    res = head(x_query, x_ref)
    loss = torch.nn.functional.binary_cross_entropy(res, targets)
    loss.backward()
    print(res.size())


if __name__ == '__main__':
    main()
