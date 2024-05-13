from pytorch_metric_learning.distances import LpDistance

from src.custom_distances.verification_similarity_base import VerificationSimilarityBase


class LpSimilarity(LpDistance, VerificationSimilarityBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_inverted = True
        self.normalize_embeddings = True
        assert self.power == 1, "Power must be 1"

    def transform_to_similarity_between_0_and_1(self, value):
        return 1. - (value / 2.)
