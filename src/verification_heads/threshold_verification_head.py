import torch
from pytorch_metric_learning.distances import BaseDistance, LpDistance, CosineSimilarity
from pytorch_metric_learning.utils.inference import MatchFinder

from src.custom_distances import LpSimilarity, VerificationCosineSimilarity, KLDivDistance
from src.verification_heads.verification_head_base import VerificationHeadBase


class ThresholdVerificationHead(VerificationHeadBase):
    def __init__(self, distance: BaseDistance, threshold=None):
        super().__init__(distance=distance, threshold=threshold)
        self._match_finder = MatchFinder(self.distance, threshold=self.best_threshold)

    def _is_normalized(self) -> bool:
        return False

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, threshold=None) -> torch.tensor:
        return torch.tensor(
            self._match_finder.get_matching_pairs(
                query_embeddings, reference_embeddings, (threshold or self.best_threshold))
        ).float()

    def get_matches(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, threshold=None, n_closest=None):
        predictions = self.forward(query_embeddings, reference_embeddings, threshold)
        return self.apply_n_closest(predictions, n_closest)


if __name__ == '__main__':
    queries = torch.randn((3, 10))
    references = torch.randn((2, 10))
    queries[:, 5:] = torch.nn.ELU()(queries[:, 5:]) + 1
    references[:, 5:] = torch.nn.ELU(alpha=0.99)(references[:, 5:]) + 1
    for klass in [LpDistance, CosineSimilarity, LpSimilarity, VerificationCosineSimilarity, KLDivDistance]:
        head = ThresholdVerificationHead(klass(), threshold=.6)
        print(head(queries, references))
    print()
    for klass in [LpDistance, CosineSimilarity, LpSimilarity, VerificationCosineSimilarity, KLDivDistance]:
        head = ThresholdVerificationHead(klass())
        print(head(queries, references, threshold=4.2))
