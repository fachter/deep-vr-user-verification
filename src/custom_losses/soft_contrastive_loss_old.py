import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import BaseMetricLossFunction, GenericPairLoss, WeightRegularizerMixin
from pytorch_metric_learning.reducers import AvgNonZeroReducer

from src.models.verification_heads import DistanceMatchProbabilityVerificationHead, MatchProbabilityHead


def loss_soft_con(probabilities: torch.tensor, ground_truth: torch.tensor):
    log_input = (1. - ground_truth) * (1. - probabilities) + ground_truth * probabilities
    clipped_inputs = torch.clip(log_input, 1e-10)
    loss = -torch.log(clipped_inputs)
    return loss


class SoftContrastivePairLoss(GenericPairLoss, WeightRegularizerMixin):
    def __init__(self, distance_match_probability_head: DistanceMatchProbabilityVerificationHead, **kwargs):
        super().__init__(**kwargs, mat_based_loss=True)
        self._match_probability_head = distance_match_probability_head.match_probability_head

    # noinspection PyMethodOverriding
    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_mask_bool = pos_mask.bool()
        neg_mask_bool = neg_mask.bool()
        anchor_positive = mat[pos_mask_bool]
        anchor_negative = mat[neg_mask_bool]

        positive_match_probabilities = self._match_probability_head(anchor_positive)
        pos_loss = loss_soft_con(positive_match_probabilities, torch.ones_like(anchor_positive))

        negative_match_probabilities = self._match_probability_head(anchor_negative)
        neg_loss = loss_soft_con(negative_match_probabilities, torch.zeros_like(anchor_negative))

        losses = pos_loss.mean() + neg_loss.mean()
        return {
            "loss": {
                "losses": losses,
                "indices": None,
                "reduction_type": "already_reduced"
            }
        }


class SoftContrastiveLoss(BaseMetricLossFunction, WeightRegularizerMixin):
    def __init__(self, distance_match_probability_head: DistanceMatchProbabilityVerificationHead, **kwargs):
        if "distance" not in kwargs:
            kwargs["distance"] = distance_match_probability_head.distance
        super().__init__(**kwargs)
        self.distance_match_probability_head: DistanceMatchProbabilityVerificationHead = distance_match_probability_head

    def compute_loss(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        gt = labels[:, None].eq(labels[None, :]).float()
        probs = self.distance_match_probability_head(embeddings, embeddings)
        loss = loss_soft_con(probs, gt).mean()

        return {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            },
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def get_default_distance(self):
        return LpDistance()


if __name__ == '__main__':
    x = torch.randn((2_000, 192))
    y = torch.randint(0, 12, (2_000,))
    head = MatchProbabilityHead()
    loss_func = SoftContrastivePairLoss(head)
    pair_loss = loss_func(x, y)
    print(pair_loss)

    pair_head = DistanceMatchProbabilityVerificationHead(head)
    loss_func = SoftContrastiveLoss(pair_head)
    total_loss = loss_func(x, y)
    print(total_loss)
