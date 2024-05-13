from pytorch_metric_learning.distances import CosineSimilarity

from src.custom_distances.verification_similarity_base import VerificationSimilarityBase


class VerificationCosineSimilarity(CosineSimilarity, VerificationSimilarityBase):

    def transform_to_similarity_between_0_and_1(self, value):
        return (value + 1.) / 2.
