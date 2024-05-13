import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.utils import common_functions as c_f, loss_and_miner_utils as lmu

from src.verification_heads import DistanceMatchProbabilityVerificationHead


def loss_soft_con(probabilities: torch.tensor, ground_truth: torch.tensor):
    log_input = (1. - ground_truth) * (1. - probabilities) + ground_truth * probabilities
    clipped_inputs = torch.clip(log_input, 1e-10)
    loss = -torch.log(clipped_inputs)
    return loss


class TripletSoftContrastiveLoss(BaseMetricLossFunction):
    def __init__(self, match_probability_verification_head: DistanceMatchProbabilityVerificationHead, **kwargs):
        super().__init__(**kwargs)
        self.match_probability_verification_head = match_probability_verification_head

    def compute_loss(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, ref_labels
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.match_probability_verification_head(embeddings, ref_emb)
        ap_probs = mat[anchor_idx, positive_idx]
        an_probs = mat[anchor_idx, negative_idx]

        ap_loss = loss_soft_con(ap_probs, torch.ones_like(ap_probs))
        an_loss = loss_soft_con(an_probs, torch.ones_like(an_probs))

        loss = ap_loss + an_loss
        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet"
            }
        }


if __name__ == '__main__':
    x = torch.randn((300, 192))
    y = torch.randint(0, 12, (300,))
    head = DistanceMatchProbabilityVerificationHead(LpDistance())
    loss_func = TripletSoftContrastiveLoss(head)
    # loss_func = TripletMarginLoss()
    lo = loss_func(x, y)
    print(lo)

