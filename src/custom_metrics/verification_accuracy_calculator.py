from typing import Dict, Tuple, Any, Set

import torch
from lightning import seed_everything
from torch import nn
from torchmetrics.classification import BinaryStatScores

from src.models.verification_heads import BaseVerificationHead, DistanceMatchProbabilityVerificationHead
from src.utils.embeddings import Embeddings


class VerificationScoreCalculator(nn.Module):
    def __init__(self, thresholds: Set[float], verification_head: BaseVerificationHead,
                 number_of_negative_users=3, return_per_class=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thresholds = [str(int(threshold * 100)) for threshold in thresholds]
        self.number_of_negative_users = number_of_negative_users
        self.stats_per_user_pair: Dict[Tuple[int, int], torch.tensor] = {}
        self.stats_per_user: Dict[(int, str), torch.tensor] = {}
        self.combined_stats: Dict[str, torch.tensor] = {}
        self.total_score_per_threshold: Dict[str, torch.tensor] = {}
        self.score_per_user_and_threshold: Dict[str, Dict[int, torch.tensor]] = {}
        self.return_per_class = return_per_class
        self._negative_users: Dict[torch.tensor, torch.tensor] = {}
        self._unique_users: torch.tensor = torch.tensor([])

        for threshold in self._thresholds:
            setattr(self, f"calculate_verification_scores_for_{threshold}",
                    lambda **_kwargs: self._verification_scores(threshold, **_kwargs))
        self.verification_head = verification_head
        self.score_functions = nn.ModuleDict({
            str(int(threshold * 100)): BinaryStatScores(
                threshold=threshold, multidim_average="global")
            for threshold in thresholds
        })

    def calculate_equal_error_rate(self):
        return {
            'EER': {
                'EER_threshold': self._equal_error_threshold,
                'EER': self._equal_error_rate
            }
        }

    def get_scores(self, query: torch.tensor, query_labels: torch.tensor,
                   reference: torch.tensor, reference_labels: torch.tensor):
        embeddings = Embeddings(
            query=query,
            reference=reference,
            query_labels=query_labels,
            reference_labels=reference_labels
        )
        self._set_unique_and_negative_users(embeddings)
        self._calculate_eer(embeddings)
        self._perform_binary_stat_score_calculation(embeddings)
        self._fill_scores()
        return self._get_scores()

    def _set_unique_and_negative_users(self, embeddings: Embeddings):
        self._unique_users = embeddings.query_labels.unique()
        self._negative_users = {
            user_id.int(): _get_random_negative_values(
                self._unique_users, user_id.int().item())[:self.number_of_negative_users]
            for user_id in self._unique_users
        }

    def _get_far_and_frr_for_threshold(self, embeddings: Embeddings, threshold: float):
        stat_threshold = threshold if self.verification_head.match_probability_head else 0.5
        binary_stat_scores = (BinaryStatScores(threshold=stat_threshold, multidim_average="global")
                              .to(embeddings.query.device))
        scores_per_user = {(ref_user.item(), ref_user.item()): binary_stat_scores(
            *self._get_matches_and_truth(embeddings, ref_user, ref_user, threshold))
            for ref_user in self._negative_users.keys()
        }
        for ref_user, query_users in self._negative_users.items():
            for query_user in query_users:
                matches, truth = self._get_matches_and_truth(embeddings, ref_user, query_user, threshold)
                scores_per_user[(ref_user.item(), query_user.item())] = binary_stat_scores(matches, truth)
        total_average = torch.cat(
            [score_per_user[None] for score_per_user in scores_per_user.values()], dim=0).sum(dim=0)
        (tp, fp, tn, fn, sup) = total_average
        far = torch.nan_to_num(fp / (fp + tn), nan=1.0)
        frr = torch.nan_to_num(fn / (fn + tp), nan=1.0)
        return far, frr

    def _get_eer(self, embeddings: Embeddings, min_threshold: float, max_threshold: float):
        threshold = (min_threshold + max_threshold) / 2.
        far, frr = self._get_far_and_frr_for_threshold(embeddings, threshold)

        if torch.isclose(far, frr, 0.001) or abs(max_threshold - min_threshold) < 1e-10:
            return threshold, frr

        far_frr = far < frr
        if self.verification_head.match_probability_head:
            far_frr = not far_frr

        if far_frr:
            return self._get_eer(embeddings, threshold, max_threshold)
        return self._get_eer(embeddings, min_threshold, threshold)

    def _calculate_eer(self, embeddings: Embeddings):
        max_threshold = 1. if self.verification_head.match_probability_head else 10_000.
        self._equal_error_threshold, self._equal_error_rate = self._get_eer(embeddings, 0., max_threshold)

    def _perform_binary_stat_score_calculation(self, embeddings: Embeddings):
        self.stats_per_user_pair: Dict[Tuple[int, int], torch.tensor] = {}
        for user in self._unique_users:
            self._fill_user_pair_stats(embeddings, user, user)

        for ref_user, neg_users in self._negative_users.items():
            for neg_user in neg_users:
                self._fill_user_pair_stats(embeddings, ref_user, neg_user)
        self._set_user_stats()

    def _fill_scores(self):
        self.total_score_per_threshold = {}
        self.score_per_user_and_threshold = {}
        self.total_score_per_threshold = {
            threshold: _get_scores_from_stats(self.combined_stats, threshold, suffix=f"_{threshold}")
            for threshold in self._thresholds
        }
        for user, threshold in self.stats_per_user.keys():
            stats = _get_scores_from_stats(self.stats_per_user, (user, threshold), suffix=f"_{threshold}/{user}")
            if threshold in self.score_per_user_and_threshold.keys():
                self.score_per_user_and_threshold[threshold][user] = stats
            else:
                self.score_per_user_and_threshold[threshold] = {user: stats}

    def _get_scores(self):
        scores = {}
        for threshold in self._thresholds:
            scores[threshold] = self._verification_scores(threshold)
        scores = scores | self.calculate_equal_error_rate()
        return scores

    def _verification_scores(self, threshold, **_kwargs):
        total_scores = self.total_score_per_threshold[threshold]
        if not self.return_per_class:
            return total_scores
        scores_for_threshold: Dict[Any, float] = self.score_per_user_and_threshold[threshold].copy()
        scores_for_threshold['total'] = total_scores
        return scores_for_threshold

    def _set_user_stats(self):
        self.stats_per_user = {}
        self.combined_stats = {}

        for key, user_score in self.stats_per_user_pair.items():
            for user_score_key, user_score_value in user_score.items():
                if user_score_key in self.combined_stats.keys():
                    self.combined_stats[user_score_key] += user_score_value
                else:
                    self.combined_stats[user_score_key] = user_score_value.clone()
                score_per_user_key = (key[0], user_score_key)
                if score_per_user_key in self.stats_per_user.keys():
                    self.stats_per_user[score_per_user_key] += user_score_value
                else:
                    self.stats_per_user[score_per_user_key] = user_score_value.clone()

    def _fill_user_pair_stats(self, embeddings: Embeddings, reference_user: torch.tensor, query_user: torch.tensor):
        matches, truth = self._get_matches_and_truth(embeddings, reference_user, query_user, self._equal_error_threshold)

        score_results = {score_key: score_function(matches, truth)
                         for score_key, score_function in self.score_functions.items()}
        self.stats_per_user_pair[((reference_user.int().item()), (query_user.int().item()))] = score_results

    def _get_matches_and_truth(self, embeddings, reference_user, query_user, threshold):
        ref = embeddings.reference[embeddings.reference_labels == reference_user]
        query_label_mask = torch.isin(embeddings.query_labels, query_user)
        que = embeddings.query[query_label_mask]
        matches = self.verification_head(que, ref, threshold)
        truth = self.verification_head.get_truth(embeddings, query_user, reference_user)

        return matches, truth


def _get_scores_from_stats(scores: Dict[Any, torch.tensor], key, suffix=''):
    tp, fp, tn, fn, sup = scores[key]
    return {
        f'precision{suffix}': (tp / (tp + fp)).item(),
        f'recall{suffix}': (tp / (tp + fn)).item(),
        f'FAR{suffix}': (fp / (fp + tn)).item(),
        f'FRR{suffix}': (fn / (fn + tp)).item()
    }


def _get_random_negative_values(unique_users: torch.tensor, number_to_exclude: int):
    unique_user_length = len(unique_users)
    min_user = unique_users.min()
    random_values = torch.randperm(unique_user_length, device=unique_users.device) + min_user
    return random_values[random_values != number_to_exclude]


if __name__ == '__main__':
    # gt = torch.tensor([1, 1, 1, 1, 1])
    # ref = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    # res = gt[:, None].eq(ref[None]).float()
    # print(gt.size())
    seed_everything(42)
    device = torch.device("cuda")
    pair_match_probability_head = DistanceMatchProbabilityVerificationHead()
    verification_evaluator = VerificationScoreCalculator(
        thresholds={.41, .4125, .415, .4175, .42, .5, .7},
        return_per_class=False,
        verification_head=pair_match_probability_head,
        number_of_negative_users=3
    ).to(device=device)
    score = verification_evaluator.score_functions["50"](torch.randn((100,)).to(device=device),
                                                         torch.randint(1, (100,)).to(device=device))
    auth_scores = verification_evaluator.get_scores(torch.randn((500, 192)).to(device=device),
                                                    torch.randint(0, 11, (500,)).to(device=device),
                                                    torch.randn((500, 192)).to(device=device),
                                                    torch.randint(0, 11, (500,)).to(device=device))

    print(auth_scores)
    # embeddings.reference_labels[embeddings.reference_labels == reference_user][:, None].eq(embeddings.query_labels[torch.isin(embeddings.query_labels, query_user)][None, :]).float()
