from typing import Optional

import torch
import torch.nn as nn
from pytorch_metric_learning.distances import BaseDistance, LpDistance

from src.utils.embeddings import Embeddings


class BaseVerificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.distance: Optional[BaseDistance] = None
        self.match_probability_head = None

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, *_args):
        raise NotImplementedError

    def get_matches_and_truth(self, embeddings: Embeddings, reference_user: int, query_user: int):
        reference = embeddings.reference[(embeddings.reference_labels == reference_user)]
        query = embeddings.query[(embeddings.query_labels == query_user)]
        matches = self._get_matches(reference, query)
        truth = self.get_truth(embeddings, query_user, reference_user)
        return matches, truth

    def get_truth(self, embeddings: Embeddings, query_user: int, reference_user: int):
        raise NotImplementedError


class MatchProbabilityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1., dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(2., dtype=torch.float32, requires_grad=True), requires_grad=True)
        # self.a = nn.Parameter(torch.rand(1, dtype=torch.float32, requires_grad=True), requires_grad=True)
        # self.b = nn.Parameter(torch.randn(1, dtype=torch.float32, requires_grad=True), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pair_distance: torch.tensor):
        return self.sigmoid(-self.a * pair_distance + self.b)


class DistanceMatchProbabilityVerificationHead(BaseVerificationHead):
    def __init__(self, match_probability_head: MatchProbabilityHead = None, distance: BaseDistance = None):
        super().__init__()
        self.match_probability_head = MatchProbabilityHead() if match_probability_head is None \
            else match_probability_head
        self.distance: BaseDistance = LpDistance(normalize_embeddings=False) if distance is None else distance

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, *_args):
        distance_matrix = self.distance(query_embeddings, reference_embeddings)
        return self.match_probability_head(distance_matrix)

    def pairwise_forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor):
        pair_distance = self.distance.pairwise_distance(query_embeddings, reference_embeddings)
        return self.match_probability_head(pair_distance)

    def get_truth(self, embeddings: Embeddings, query_user: int, reference_user: int):
        query_mask = torch.isin(embeddings.query_labels, query_user)
        return embeddings.query_labels[query_mask][:, None].eq(
            embeddings.reference_labels[embeddings.reference_labels == reference_user][None]).float()


class NClosestThresholdVerificationHead(BaseVerificationHead):
    def __init__(self, threshold: torch.tensor = None, n: int = 100, distance: BaseDistance = None):
        super().__init__()
        self.distance: BaseDistance = LpDistance(normalize_embeddings=False) if distance is None else distance
        self.n = n
        self.threshold = torch.tensor(0.5) if threshold is None else threshold

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, threshold=None, *_args):
        threshold = self.threshold if threshold is None else threshold
        distance_matrix = self.distance(query_embeddings, reference_embeddings)
        smallest, _ = torch.topk(distance_matrix, min(self.n, distance_matrix.size(1)), largest=False)
        return smallest.le(threshold).float().sum(dim=1) / smallest.size(1)

    def get_truth(self, embeddings: Embeddings, query_user: int, reference_user: int):
        return embeddings.query_labels[embeddings.query_labels == query_user].eq(reference_user).float()


def _mahalanobis(query: torch.tensor, references: torch.tensor):
    mean = references.mean(dim=0)
    x_mu = query - mean
    cov_matrix = torch.cov(references.T)
    inverse_cov = torch.inverse(cov_matrix)
    left_term = x_mu @ inverse_cov
    full_term = left_term @ x_mu.T
    return torch.sqrt(full_term.diagonal() if full_term.dim() == 2 else full_term)


class MahalanobisThresholdVerificationHead(BaseVerificationHead):
    def __init__(self, threshold: torch.tensor = None, distance: LpDistance = None):
        super().__init__()
        self.threshold = torch.tensor(0.5) if threshold is None else threshold
        self.distance = LpDistance(normalize_embeddings=False) if distance is None else distance

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, threshold=None, *_args):
        threshold = self.threshold if threshold is None else threshold
        mahalanobis_distance = _mahalanobis(query_embeddings, reference_embeddings)
        return mahalanobis_distance.le(threshold).float()

    def get_truth(self, embeddings: Embeddings, query_user: int, reference_user: int):
        return embeddings.query_labels[embeddings.query_labels == query_user].eq(reference_user).float()


# class MahalanobisMatchProbabilityVerificationHead(BaseVerificationHead):
#     def __init__(self, match_probability_head: MatchProbabilityHead = None):
#         super().__init__()
#         self.match_probability_head = MatchProbabilityHead() if match_probability_head is None\
#             else match_probability_head
#
#     def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, *_args):
#         mahalanobis_distance = _mahalanobis(query_embeddings, reference_embeddings)
#         return self.match_probability_head(mahalanobis_distance)
#
#     def get_truth(self, embeddings: Embeddings, query_user: int, reference_user: int):
#         return embeddings.query_labels[embeddings.query_labels == query_user].eq(reference_user).float()

