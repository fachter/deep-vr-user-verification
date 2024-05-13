import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import GenericPairLoss, BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f, loss_and_miner_utils as lmu

from src.custom_distances import KLDivDistance
from src.custom_losses.helper import loss_soft_con
from src.verification_heads import DistanceMatchProbabilityVerificationHead
from src.verification_heads.sampling_match_probability_verification_head import SamplingMatchProbabilityVerificationHead


class PairSoftContrastiveLoss(BaseMetricLossFunction):
    def __init__(self, match_probability_verification_head: DistanceMatchProbabilityVerificationHead, **kwargs):
        super().__init__(**kwargs)
        self.match_probability_verification_head = match_probability_verification_head

    def compute_loss(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_pairs(
            indices_tuple, labels, ref_labels
        )
        positive_anchor_idx, positive_idx, negative_anchor_idx, negative_idx = indices_tuple
        pos_loss, neg_loss = 0, 0
        if len(positive_anchor_idx) > 0:
            pos_probs = self.match_probability_verification_head.pairwise_forward(embeddings[positive_anchor_idx],
                                                                                  ref_emb[positive_idx])
            pos_loss = loss_soft_con(pos_probs, torch.ones_like(pos_probs))

        if len(negative_anchor_idx) > 0:
            neg_probs = self.match_probability_verification_head.pairwise_forward(embeddings[negative_anchor_idx],
                                                                                  ref_emb[negative_idx])
            neg_loss = loss_soft_con(neg_probs, torch.zeros_like(neg_probs))

        pos_pairs = lmu.pos_pairs_from_tuple(indices_tuple)
        neg_pairs = lmu.neg_pairs_from_tuple(indices_tuple)

        return {
            "pos_loss": {
                "losses": pos_loss,
                "indices": pos_pairs,
                "reduction_type": "pos_pair"
            },
            "neg_loss": {
                "losses": neg_loss,
                "indices": neg_pairs,
                "reduction_type": "neg_pair"
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def _sub_loss_names(self):
        return ["pos_loss", "neg_loss"]


if __name__ == '__main__':
    x = torch.randn((300, 192))
    y = torch.randint(0, 12, (300,))
    head = DistanceMatchProbabilityVerificationHead(LpDistance())
    loss_func = PairSoftContrastiveLoss(head)
    # loss_func = TripletMarginLoss()
    lo = loss_func(x, y)
    print(lo)
    head_kl_div = SamplingMatchProbabilityVerificationHead(KLDivDistance())
    kl_div_loss = PairSoftContrastiveLoss(head_kl_div)
    lo2 = loss_func(x, y)
    print(lo2)
