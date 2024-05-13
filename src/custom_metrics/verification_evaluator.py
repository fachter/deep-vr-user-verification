from typing import List

import pandas as pd
import torch
import torch.nn as nn
import wandb
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity
from torchmetrics.classification import BinaryROC, BinaryStatScores
from tqdm import tqdm

from src.utils.embeddings import Embeddings
from src.verification_heads import VerificationHeadBase, SimilarityVerificationHead, ThresholdVerificationHead
from src.verification_heads.mahalanobis_verification_head import MahalanobisVerificationHead


def _get_n_closest_predictions_and_gt(predictions, gt, reshaped_references, n_closest):
    prediction_per_n = []
    gt_per_n = []
    for user in reshaped_references.unique():
        user_mask = reshaped_references == user
        user_predictions = predictions[:, user_mask]
        user_gt = gt[:, user_mask]
        largest, largest_indices = torch.topk(
            user_predictions, min(n_closest, user_predictions.size(1)), largest=True)
        prediction_per_n.append(largest)
        gt_per_n.append(torch.gather(user_gt, 1, largest_indices))
    all_predictions = torch.concat(prediction_per_n, dim=1).flatten()
    all_gts = torch.concat(gt_per_n, dim=1).flatten()
    return all_predictions, all_gts


class VerificationEvaluator(nn.Module):
    def __init__(self, verification_head: VerificationHeadBase, far_threshold: List[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._verification_head = verification_head
        self._far_threshold = [1e-2, 1e-3, 1e-4, 1e-5] if far_threshold is None else far_threshold

    def get_scores(self, embeddings: Embeddings):
        predictions, gt, reshaped_references = self._verification_head.compare_groupwise(embeddings)
        scores_per_n = dict()
        for n_closest in tqdm(self._verification_head.n_closest):
            all_predictions, all_gts = _get_n_closest_predictions_and_gt(predictions, gt, reshaped_references,
                                                                         n_closest)

            broc = BinaryROC()
            fpr, tpr, thresholds = broc(all_predictions, all_gts)
            fnr = 1 - tpr
            index_for_eer = torch.argmin(torch.absolute(fnr - fpr))
            eer = fpr[index_for_eer]
            far = fpr
            threshold_indices = {f"TAR@FAR={far_threshold * 100}%": torch.argmin(torch.abs(far - far_threshold))
                                 for far_threshold in self._far_threshold}
            scores = dict()
            far_threshold_values = dict()
            for key, index in threshold_indices.items():
                scores[f"TAR/{n_closest}/{key}"] = tpr[index]
                thresholds_value = thresholds[index]
                if not self._verification_head.is_normalized:
                    thresholds_value = self._verification_head.denormalize_values(thresholds_value)
                far_threshold_values[f"TAR_threshold/{n_closest}/{key}"] = thresholds_value
            scores['FAR_thresholds'] = far_threshold_values
            scores[f'EER/{n_closest}'] = eer
            scores[f'EER_threshold/{n_closest}'] = thresholds[index_for_eer]
            scores_per_n[n_closest] = scores
        eer_index = -2 if len(self._verification_head.n_closest) > 1 else -1
        second_last_n = self._verification_head.n_closest[eer_index]
        main_eer = scores_per_n[second_last_n][f'EER/{second_last_n}']
        main_eer_threshold = scores_per_n[second_last_n][f'EER/{second_last_n}']
        scores_per_n[second_last_n]['EER'] = main_eer
        scores_per_n[second_last_n]['EER_threshold'] = main_eer_threshold
        return scores_per_n

    def test_mode(self, embeddings: torch.tensor, take_ids: torch.tensor,
                  reference_masks: torch.tensor, query_masks: torch.tensor, log_to_wandb=True):
        """
        reference_mask: 2D tensor (1st dimension = user, 2nd dim = takes),
        e.g. (10, 20) a list of 10 users with 20 takes each
        query_mask: 3D tensor (1st dimension = user, 2nd dim = number of queries, 3rd dim = takes),
        e.g. (10, 100, 2) a list of 10 users with 100 query tries, each using embeddings for 2 takes
        """
        result_per_n = {}
        for n_closest in self._verification_head.n_closest:
            result_matrix = torch.zeros((reference_masks.size(0), query_masks.size(0)))
            for ref_user, reference_take_mask in tqdm(enumerate(reference_masks), desc="Reference users"):
                reference_mask = torch.isin(take_ids, reference_take_mask)
                reference_embeddings = embeddings[reference_mask]
                for query_user, queries_per_user in enumerate(query_masks):
                    pair_result = []

                    for query_set_index, query_per_user_mask in enumerate(queries_per_user):
                        query_mask = torch.isin(take_ids, query_per_user_mask)
                        query_embeddings = embeddings[query_mask]
                        result = self._verification_head.get_matches(
                            query_embeddings, reference_embeddings, n_closest=n_closest
                        )
                        pair_result.append(result.mean())
                    result_matrix[query_user][ref_user] = torch.stack(pair_result).mean()
            result_per_n[n_closest] = result_matrix
            df = pd.DataFrame(result_matrix)
            result_table = wandb.Table(dataframe=df)
            if log_to_wandb:
                wandb.log({f'match_predictions/at_{n_closest}': result_table})

        for n, result in result_per_n.items():
            df = pd.DataFrame(result)
            result_table = wandb.Table(dataframe=df)
            if log_to_wandb:
                wandb.log({f'match_predictions/at_{n}': result_table})

        return result_per_n


    def _get_test_predictions(self, embeddings: Embeddings):
        result_per_n = {}
        for n_closest in self._verification_head.n_closest:
            res = torch.zeros((len(embeddings.query_labels.unique()), len(embeddings.reference_labels.unique())))
            for q_idx, query_user in enumerate(embeddings.query_labels.unique()):
                for r_idx, reference_user in enumerate(embeddings.reference_labels.unique()):
                    predictions = self._verification_head.get_matches(
                        embeddings.query[embeddings.query_labels == query_user],
                        embeddings.reference[embeddings.reference_labels == reference_user],
                        n_closest=n_closest
                    )
                    res[q_idx, r_idx] = predictions.mean()

            result_per_n[n_closest] = res
        return result_per_n


def main():
    embeddings = Embeddings(
        query=torch.randn((10_000, 128)),
        query_labels=torch.randint(0, 9, (10_000,)),
        reference=torch.randn((5_000, 128)),
        reference_labels=torch.randint(0, 9, (5_000,))
    )
    scores_per_klass = dict()
    # for head_klass in [MahalanobisVerificationHead, SimilarityVerificationHead, ThresholdVerificationHead]:
    for index, head in enumerate([MahalanobisVerificationHead(threshold=50.), SimilarityVerificationHead(threshold=0.3),
                                  ThresholdVerificationHead(LpDistance(), threshold=0.5),
                                  ThresholdVerificationHead(CosineSimilarity(), threshold=-0.1)]):
        evaluator = VerificationEvaluator(head)
        scores = evaluator.test_mode(embeddings)
        scores_per_klass[index] = scores

    print("Done")


if __name__ == '__main__':
    main()
