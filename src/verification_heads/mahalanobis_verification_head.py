from typing import List

import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils.inference import FaissKMeans, MatchFinder

from src.utils.embeddings import Embeddings
from src.verification_heads.verification_head_base import VerificationHeadBase
from src.custom_distances.mahalanobis_distance import mahalanobis_distance


class MahalanobisVerificationHead(VerificationHeadBase):
    def __init__(self, distance=LpDistance(), k_cluster=5, n_closest: List[int] = None, threshold=None):
        super().__init__(distance, threshold=threshold)
        assert isinstance(distance, LpDistance) and distance.p == 2, "Mahalanobis only works with L2 Distance"
        self._k = k_cluster
        self._n_closest_range = list(
            range(1, self._k + 1)) if n_closest is None or n_closest == "None" else n_closest or len(n_closest) == 0

    def _is_normalized(self) -> bool:
        return False

    def _n_closest(self):
        return self._n_closest_range

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, threshold=None):
        t = threshold or self.best_threshold
        return self.forward_distance(query_embeddings, reference_embeddings).le(t).float()

    def get_matches(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor,
                    threshold=None, n_closest=None) -> torch.tensor:
        predictions = self.forward(query_embeddings, reference_embeddings, threshold)
        return self.apply_n_closest(predictions, n_closest)

    def forward_distance(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor) -> torch.tensor:
        k_means_func = FaissKMeans()
        clusters = k_means_func(reference_embeddings, self._k)
        cluster_distances = torch.full((query_embeddings.size(0), self._k), torch.inf)
        for cluster in range(self._k):
            dist = mahalanobis_distance(query_embeddings, reference_embeddings[clusters == cluster])
            cluster_distances[:, cluster] = dist
        try:
            cluster_distances[cluster_distances.isnan()] = cluster_distances[~cluster_distances.isnan()].max()
        except RuntimeError:
            cluster_distances[cluster_distances.isnan()] = (self.best_threshold or 1_000_000_000) + 1.
        return cluster_distances


def main():
    head = MahalanobisVerificationHead()
    dim = 2
    query = torch.randn((100, dim))
    reference = torch.randn((400, dim))
    query_labels = torch.randint(0, 9, (100,))
    reference_labels = torch.randint(0, 9, (400,))
    embeddings = Embeddings(
        query=query,
        query_labels=query_labels,
        reference=reference,
        reference_labels=reference_labels,
    )
    prediction, gt, reshaped_refs = head.compare_groupwise(embeddings)
    print(prediction.size())
    print(gt.size())
    user_mask = reshaped_refs == 3
    print(prediction[:, user_mask].size())
    print(gt[:, user_mask].size())


if __name__ == '__main__':
    main()
