import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f, loss_and_miner_utils as lmu

from src.custom_distances import KLDivDistance
from src.custom_losses.helper import loss_soft_con
from src.verification_heads import SamplingMatchProbabilityVerificationHead


class HibPairLoss(BaseMetricLossFunction):
    def __init__(self, match_probability_verification_head: SamplingMatchProbabilityVerificationHead, beta, **kwargs):
        super().__init__(**kwargs)
        self._kl_div = KLDivDistance()
        self._match_probability_head = match_probability_verification_head
        self._beta = beta

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_pairs(
            indices_tuple, labels, ref_labels
        )
        positive_anchor_idx, positive_idx, negative_anchor_idx, negative_idx = indices_tuple
        pos_loss, neg_loss = 0, 0
        if len(positive_anchor_idx) > 0:
            pos_loss = self.vib_emb_loss(embeddings[positive_anchor_idx], ref_emb[positive_idx], True)

        if len(negative_anchor_idx) > 0:
            neg_loss = self.vib_emb_loss(embeddings[negative_anchor_idx], ref_emb[negative_idx], False)

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

    def vib_emb_loss(self, embeddings: torch.tensor, ref_emb: torch.tensor, positive_pairs: bool):
        pos_probs = self._match_probability_head.pairwise_forward(embeddings, ref_emb)
        dim = embeddings.size(1) // 2
        ground_truth = torch.ones_like(pos_probs) if positive_pairs else torch.zeros_like(pos_probs)
        soft_con_loss = loss_soft_con(pos_probs, ground_truth)
        kl_regularisation = self._beta * (
                self._kl_div(embeddings, torch.tensor([[0.] * dim + [1.] * dim], device=embeddings.device)) +
                self._kl_div(ref_emb, torch.tensor([[0.] * dim + [1.] * dim],device=ref_emb.device)))
        return soft_con_loss.reshape(-1, 1) + kl_regularisation


def main():
    device = torch.device("cuda")
    queries = torch.tensor([
        [10.2, 5., 2.2, 0.2],
        [10., 5.2, 2.2, 0.2],
        [9.9, 4.8, 2.2, 0.2],
        [-3., 3.2, 2.2, 0.2],
        [-3.1, 2.9, 2.2, 0.2],
        [-3.1, 3., 2.2, 0.2],
        [0., 0., 2.2, 0.2],
        [0.1, 0.1, 2.2, 0.2],
        [0.2, 0., 2.2, 0.2]
    ], device=device)
    labels = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3], device=device)
    head = SamplingMatchProbabilityVerificationHead(LpDistance()).to(device)
    loss_func = HibPairLoss(head, 0.5).to(device)
    loss = loss_func(queries, labels)
    print(loss)
    loss.backward()


if __name__ == '__main__':
    main()
