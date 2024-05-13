from typing import Optional
import torch
import torch.nn as nn
from pytorch_metric_learning.distances import BaseDistance

from src.utils.embeddings import Embeddings


class VerificationHeadBase(nn.Module):

    def __init__(self, distance=None, threshold=None):
        super().__init__()
        self.distance: Optional[BaseDistance] = distance
        self.min_value = None
        self.max_value = None
        self.best_threshold = threshold

    @property
    def is_normalized(self) -> bool:
        return self._is_normalized()

    @property
    def n_closest(self) -> list:
        return self._n_closest()

    def _is_normalized(self) -> bool:
        raise NotImplementedError

    def _n_closest(self):
        return [1, 5, 10, 50, 100, torch.inf]

    def forward(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, *_args) -> torch.tensor:
        raise NotImplementedError

    def get_matches(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor, threshold=None, n_closest=None) -> torch.tensor:
        raise NotImplementedError

    def apply_n_closest(self, predictions, n_closest):
        if n_closest is None:
            return predictions
        largest, _ = torch.topk(predictions, min(n_closest, predictions.size(1)), largest=True)
        return largest

    def forward_distance(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor) -> torch.tensor:
        return self._distance_forward_in_batches(query_embeddings, reference_embeddings)

    def compare_groupwise(self, embeddings: Embeddings) -> torch.tensor:
        all_user_results = []
        all_ground_truths = []
        all_reshaped_references = []
        for user in embeddings.reference_labels.unique():
            user_reference_mask = embeddings.reference_labels == user
            result_per_user = self.forward_distance(
                embeddings.query,
                embeddings.reference[user_reference_mask]
            )
            reference_user = torch.tensor([user] * result_per_user.size(1))
            user_gt = embeddings.query_labels[:, None].eq(reference_user[None]).long()
            all_user_results.append(result_per_user)
            all_ground_truths.append(user_gt)
            all_reshaped_references.append(reference_user)
        results = torch.concat(all_user_results, dim=-1)
        gts = torch.concat(all_ground_truths, dim=-1)
        reshaped_references = torch.concat(all_reshaped_references, dim=-1)
        if not self.is_normalized:
            results = self.normalize_distance_to_similarity(results)
        return results, gts, reshaped_references

    def _distance_forward_in_batches(self, query_embeddings: torch.tensor, reference_embeddings: torch.tensor,
                                     block_batch_size: int = 1_000) -> torch.tensor:
        n, m = query_embeddings.size(0), reference_embeddings.size(0)

        if query_embeddings.dim() == 3:
            assert query_embeddings.size(1) == reference_embeddings.size(1), "query and ref shapes don't match"
            result_matrix = torch.zeros((n, m, query_embeddings.size(1) * reference_embeddings.size(1)))
        else:
            result_matrix = torch.zeros(n, m)
        result_matrix = result_matrix.to(device=query_embeddings.device)

        for i in range(0, n, block_batch_size):
            for j in range(0, m, block_batch_size):
                if query_embeddings.dim() == 3:
                    k_index = 0
                    for k_q in range(query_embeddings.size(1)):
                        for k_r in range(reference_embeddings.size(1)):
                            block_query = query_embeddings[i:i + block_batch_size, k_q]
                            block_references = reference_embeddings[j:j + block_batch_size, k_r]

                            block_result = self.distance(block_query, block_references)

                            result_matrix[i:i + block_batch_size, j:j + block_batch_size, k_index] = block_result
                            k_index += 1
                else:
                    block_query = query_embeddings[i:i + block_batch_size]
                    block_references = reference_embeddings[j:j + block_batch_size]

                    block_result = self.distance(block_query, block_references)

                    result_matrix[i:i + block_batch_size, j:j + block_batch_size] = block_result
        return result_matrix

    def normalized_to_similarity(self, query_embeddings: torch.tensor,
                                 reference_embeddings: torch.tensor) -> torch.tensor:
        distance_mat = self._distance_forward_in_batches(query_embeddings, reference_embeddings)
        return self.normalize_distance_to_similarity(distance_mat)

    def normalize_distance_to_similarity(self, distances: torch.tensor):
        self.min_value = distances[~distances.isnan()].min()
        self.max_value = distances[~distances.isnan()].max()
        normalized_distances = (distances - self.min_value) / (self.max_value - self.min_value)
        if not self.distance.is_inverted:
            normalized_distances = 1 - normalized_distances
        normalized_distances = normalized_distances.nan_to_num(nan=0.)
        return normalized_distances

    def denormalize_values(self, values: torch.tensor) -> torch.tensor:
        if not self.distance.is_inverted:
            values = 1 - values
        return values * (self.max_value - self.min_value) + self.min_value
