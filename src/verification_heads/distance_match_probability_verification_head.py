import torch
from pytorch_metric_learning.distances import BaseDistance, LpDistance, CosineSimilarity
from torch import nn

from src.utils.embeddings import Embeddings
from src.custom_distances import LpSimilarity, VerificationCosineSimilarity, KLDivDistance
from src.verification_heads.verification_head_base import VerificationHeadBase


class DistanceMatchProbabilityVerificationHead(VerificationHeadBase):
    def __init__(self, distance: BaseDistance, threshold=None):
        super().__init__(distance=distance, threshold=threshold)
        self.a = nn.Parameter(torch.rand(1, dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def _is_normalized(self) -> bool:
        return True

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, *_args):
        distance_matrix = self._distance_forward_in_batches(query_embeddings, reference_embeddings)
        return self._match_probability(distance_matrix)

    def get_matches(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor,
                    threshold=None, n_closest=None) -> torch.tensor:
        predictions = self.forward(query_embeddings, reference_embeddings).ge((threshold or self.best_threshold)).float()
        return self.apply_n_closest(predictions, n_closest)

    def _match_probability(self, distance_matrix):
        return self.sigmoid(-self.a * distance_matrix + self.b)

    def forward_with_distances(self, distances):
        return self._match_probability(distances)

    def pairwise_forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor):
        distances = []
        for batch in range(0, len(query_embeddings), 1_000):
            distance = self.distance.pairwise_distance(
                query_embeddings[batch:batch + 1_000], reference_embeddings[batch:batch + 1_000]
            )
            distances.append(distance)
        pair_distance = torch.concat(distances, dim=0)
        return self._match_probability(pair_distance)

    def get_truth(self, embeddings: Embeddings, query_user: int, reference_user: int, pairwise_truth=False):
        if pairwise_truth:
            query_mask = torch.isin(embeddings.query_labels, query_user)
            return embeddings.query_labels[query_mask].eq(
                embeddings.reference_labels[embeddings.reference_labels == reference_user]
            ).float()
        return super().get_truth(embeddings, query_user, reference_user)


if __name__ == '__main__':
    queries = torch.randn((10, 10))
    references = torch.randn((8, 10))
    query_labels = torch.FloatTensor([0] * 4 + [1] * 4 + [2] * 2)
    reference_labels = torch.FloatTensor([0] * 4 + [1] * 4)
    queries[:, 5:] = torch.nn.ELU()(queries[:, 5:]) + 1
    references[:, 5:] = torch.nn.ELU(alpha=0.99)(references[:, 5:]) + 1
    embeddings = Embeddings(
        queries,
        references,
        query_labels,
        reference_labels
    )
    for klass in [LpDistance, CosineSimilarity, LpSimilarity, VerificationCosineSimilarity, KLDivDistance]:
        head = DistanceMatchProbabilityVerificationHead(klass())
        print(head.get_truth(embeddings, 1, 0, True))
