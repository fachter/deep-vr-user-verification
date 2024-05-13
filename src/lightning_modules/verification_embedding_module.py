import random
import time
from typing import Any, List, Dict

import lightning as L
import numpy as np
import pytorch_metric_learning.losses
import torch
import wandb
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_metric_learning.distances import BaseDistance
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.calibration import LabelEncoder
from torch import nn

from src.custom_metrics.evaluators import Evaluators
from src.custom_metrics.verification_evaluator import VerificationEvaluator
from src.utils import utils
from src.custom_distances.kl_divergence_distance import KLDivDistance
import src.custom_losses
from src.utils.embeddings import Embeddings
from src.verification_heads import (VerificationHeadBase, DistanceMatchProbabilityVerificationHead,
                                    SimilarityVerificationHead, ThresholdVerificationHead,
                                    SamplingMatchProbabilityVerificationHead, MahalanobisVerificationHead)

logger = utils.get_logger(__name__)


class VerificationEmbeddingModule(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            loss_weights=None,
            optimizer_options: dict = None,
            metrics: List[dict] = None,
            klasses: List[str] = None,
            loss_options: Dict[str, Any] = None,
            miner_options: Dict[str, Any] = None,
            additional_options: Dict[str, Any] = None,
            verification_head: VerificationHeadBase = None
    ):
        super().__init__()
        self.best_logged_metrics = {}
        self.klasses = klasses
        self.num_classes = len(klasses)
        self.optimizer_options = {} if optimizer_options is None else optimizer_options
        self.hyperparameters = model.hparams
        self.model = model

        self.batch_tuning_mode = False
        # self.validation_step_outputs = []
        self.validation_step_predictions = []
        self.validation_step_targets = []
        self.validation_step_take_ids = []

        self.test_step_predictions = []
        self.test_step_targets = []
        self.test_step_take_ids = []

        if loss_weights is not None:
            self.loss_weights = torch.from_numpy(loss_weights).float().to(self.device)
        else:
            self.loss_weights = None

        additional_options = {} if additional_options is None else additional_options
        self._initialize_verification_head(verification_head)

        self._initialize_loss_function(loss_options)
        device = torch.device("cpu")
        self.evaluators = Evaluators(
            verification=VerificationEvaluator(self.verification_head).to(device=device),
            identification=AccuracyCalculator(device=device)
        )

        if wandb.run is not None:
            wandb.define_metric("precision_at_1/validation", summary="max")
            wandb.define_metric("EER/1/validation", summary="min")
            wandb.define_metric("EER/validation", summary="min")

            # wandb.define_metric("per_take_accuracy/validation", summary="max")
            # wandb.define_metric("per_user_accuracy/validation", summary="max")
            wandb.define_metric("loss_train", summary="min")
        self.variance_activation = nn.ELU(alpha=0.99) if self.probabilistic_embeddings else None
        self.save_hyperparameters()

    def _initialize_verification_head(self, verification_head: VerificationHeadBase):
        if verification_head is None:
            verification_head = DistanceMatchProbabilityVerificationHead(KLDivDistance())
        self.verification_head: VerificationHeadBase = verification_head
        self.distance: BaseDistance = self.verification_head.distance

    def _initialize_loss_function(self, loss_options):
        loss_options: Dict[str, Any] = (
            {
                "name": "TripletSoftContrastiveLoss",
                "custom_loss": True
            }
            if loss_options is None
            else loss_options
        )
        loss_options = loss_options.copy()

        if loss_options.get("num_classes") == "auto":
            loss_options["num_classes"] = self.num_classes
        if loss_options.get("embedding_size") == "auto":
            loss_options["embedding_size"] = self.model.num_out_classes
        if loss_options.pop("custom_loss", None):
            assert isinstance(self.verification_head, DistanceMatchProbabilityVerificationHead), (
                "SoftContrastiveLoss only works with DistanceMatchProbabilityVerificationHead"
            )
            loss_options["match_probability_verification_head"] = self.verification_head
            source = src.custom_losses
        else:
            source = pytorch_metric_learning.losses
        if "distance" not in loss_options:
            loss_options["distance"] = self.distance

        self.loss_func = getattr(source, loss_options.pop("name"))(
            **loss_options
        )

    @property
    def probabilistic_embeddings(self):
        return (isinstance(self.distance, KLDivDistance)
                or isinstance(self.verification_head, SamplingMatchProbabilityVerificationHead))

    def forward(self, x):
        output = self.model.forward(x.float())
        if not self.probabilistic_embeddings:
            return output
        cut = output.size(1) // 2
        means = output[:, :cut]
        variance = self.variance_activation(output[:, cut:]) + 1
        gaussian_output = torch.cat([means, variance], 1)
        return gaussian_output

    def do_comparison(self, x_query, x_ref):
        query_embeddings = self.forward(x_query)
        ref_embeddings = self.forward(x_ref)

        probabilities = self.verification_head(query_embeddings, ref_embeddings)
        return probabilities

    def training_step(self, batch, batch_idx):
        x = batch["data"]
        y = batch["targets"]

        embedding = self.forward(x)
        loss = self.loss_func(embedding, y.long())

        self._log_train_step(loss)

        return loss

    def _log_train_step(self, loss):
        self.log(
            f"loss_train", loss, on_step=True, on_epoch=True
        )
        if isinstance(self.verification_head, DistanceMatchProbabilityVerificationHead):
            self.log(
                "match_probability_head/a", self.verification_head.a
            )
            self.log(
                "match_probability_head/b", self.verification_head.b
            )

    def configure_optimizers(self):
        optimizer_name = self.optimizer_options.pop("name", "Adam")
        verification_head_params = list(self.verification_head.parameters())
        optimizer = getattr(torch.optim, optimizer_name)(
            params=list(self.model.parameters()) + verification_head_params,
            **self.optimizer_options
        )

        return optimizer

    def validation_step(self, batch, _batch_idx):
        X = batch["data"]
        y = batch["targets"]
        take_ids = batch["take_id"]
        # session_index = batch['session_idx']

        h = self.forward(X)
        # self.validation_step_predictions.append(h)
        self.validation_step_predictions.append(h)
        self.validation_step_targets.append(y)
        self.validation_step_take_ids.append(take_ids)
        # self.validation_step_take_ids.append(take_ids)
        # self.validation_step_outputs.append((h, y, take_ids))

    def test_step(self, batch, _batch_idx):
        X = batch["data"]
        y = batch["targets"]
        take_ids = batch["take_id"]

        h = self.forward(X)
        self.test_step_predictions.append(h)
        self.test_step_targets.append(y)
        self.test_step_take_ids.append(take_ids)

    def predict_step(self, batch, batch_idx, **kwargs):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        if self.batch_tuning_mode:
            return

        embeddings = torch.cat(self.validation_step_predictions).detach().cpu()
        y = torch.cat(self.validation_step_targets).detach().cpu()
        # session_indices = torch.cat(self.validation_step_session_indices).detach().cpu()
        take_ids = torch.Tensor(
            LabelEncoder().fit_transform(
                [take_id for take_id_batch in self.validation_step_take_ids for take_id in take_id_batch]
            )
        )

        # self._compute_and_log_validation_metrics(embeddings, y, session_indices)
        self._compute_and_log_validation_metrics_with_takes(embeddings, y, take_ids)
        self._note_best_metric_values()

        # self.validation_step_outputs = []
        self.validation_step_predictions = []
        self.validation_step_targets = []
        self.validation_step_take_ids = []
        self.validation_step_take_ids = []

    def _compute_and_log_validation_metrics_with_takes(self, embeddings, y, take_ids: torch.Tensor):
        torch.set_num_threads(8)

        n_reference_takes = 100
        n_query_takes = 200
        num_users = len(torch.unique(y))

        reference_takes_list = []
        query_takes_list = []

        for user_id in torch.unique(y):
            user_takes = torch.unique(take_ids[y == user_id])
            random.shuffle(user_takes)
            reference_takes_list.append(user_takes[:n_reference_takes])
            query_takes_list.append(user_takes[n_reference_takes:n_reference_takes + n_query_takes])

        reference_takes = torch.cat(reference_takes_list)
        query_takes = torch.cat(query_takes_list)

        assert len(reference_takes) == num_users * n_reference_takes, (f"{len(reference_takes)} != "
                                                                       f"{num_users} * {n_reference_takes}")
        assert len(query_takes) == num_users * n_query_takes, (f"{len(query_takes)} != "
                                                               f"{num_users} * {n_query_takes}")

        reference_mask = torch.isin(take_ids, reference_takes)
        query_mask = torch.isin(take_ids, query_takes)

        reference_embeddings = embeddings[reference_mask]
        reference_y = y[reference_mask]
        query_embeddings = embeddings[query_mask]
        query_y = y[query_mask]

        logger.info("Starting to compute validation metrics with "
                    f"{len(reference_y)} references and {len(query_y)} queries")
        if query_embeddings.any() and reference_embeddings.any():
            try:
                start = time.time()
                with torch.no_grad():
                    with torch.cuda.device(-1):
                        with torch.autocast(enabled=False, device_type="cpu"):
                            accuracy = self.evaluators.identification.get_accuracy(
                                query_embeddings.contiguous(),
                                query_y.contiguous(),
                                reference_embeddings.contiguous(),
                                reference_y.contiguous(),
                                ref_includes_query=False,
                            )
                            end_identification = time.time()
                            logger.info(f"Finished identification accuracy calculation "
                                        f"in {end_identification - start:.3f} s")
                            self.evaluators.verification.to(query_embeddings.device)
                            verification_scores = self.evaluators.verification.get_scores(
                                Embeddings(
                                    query=query_embeddings,
                                    query_labels=query_y,
                                    reference=reference_embeddings,
                                    reference_labels=reference_y
                                )
                            )
                            self.evaluators.verification.to(self.device)
                            end = time.time()
                            logger.info("Finished verification accuracy calculation "
                                        f"in {end - end_identification:.3f}")
                            # probs = self.match_probability_head(query_embeddings, reference_embeddings)
                            # ground_truth = query_y[:, None].eq(reference_y[None, :])
                            # pr = wandb.plots.precision_recall(ground_truth, probs)
                self.log_metrics(accuracy, "validation")
                self.log_metrics(verification_scores, "validation")
                self.log("validation_computation_time", end - start)
                self.log("validation_identification_compute_time", end_identification - start)
                self.log("validation_verification_compute_time", end - end_identification)
                # self.log('PR', pr)
            except RuntimeError as e:
                logger.exception(f"error while computing validation metrics: {e}")

    def _note_best_metric_values(self):
        for metric_name, value in self.trainer.logged_metrics.items():
            old_min_value = self.best_logged_metrics.get((metric_name, "min"), np.inf)
            old_max_value = self.best_logged_metrics.get((metric_name, "max"), -np.inf)

            self.best_logged_metrics[(metric_name, "min")] = min(
                old_min_value, self.trainer.logged_metrics[metric_name]
            )
            self.best_logged_metrics[(metric_name, "max")] = max(
                old_max_value, self.trainer.logged_metrics[metric_name]
            )

    def log_metrics(self, metrics: dict, stage: str):
        """Logs the metrics recursively."""
        for key, item in metrics.items():
            if not isinstance(item, dict):
                name = f"{key}/{stage}"
                self.log(name, item)
            else:
                self.log_metrics(item, stage)

    def count_total_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def on_test_epoch_end(self) -> None:
        embeddings = torch.cat(self.test_step_predictions)
        y = torch.cat(self.test_step_targets)
        take_ids = torch.Tensor(
            LabelEncoder().fit_transform(
                [take_id for take_id_batch in self.test_step_take_ids for take_id in take_id_batch]
            )
        ).to(embeddings.device)

        self._compute_and_log_validation_metrics_with_takes(
            embeddings.detach().cpu()[::5], y.detach().cpu()[::5], take_ids.detach().cpu()[::5]
        )
        if (not isinstance(self.verification_head, MahalanobisVerificationHead)
                and not isinstance(self.verification_head, SamplingMatchProbabilityVerificationHead)):
            self._compute_and_log_test_metrics(embeddings, y, take_ids)
        self._note_best_metric_values()

        self.test_step_predictions = []
        self.test_step_targets = []
        self.test_step_take_ids = []
        self.test_step_take_ids = []

    def _compute_and_log_test_metrics(self, embeddings: torch.tensor, y: torch.tensor, take_ids: torch.tensor):
        query_takes, reference_takes = self._get_query_and_reference_takes_for_testing(embeddings, y, take_ids)

        self.evaluators.verification.test_mode(embeddings, take_ids, reference_takes, query_takes)

    def _get_query_and_reference_takes_for_testing(self, embeddings, y, take_ids):
        n_reference_takes = 50
        n_query_takes = 200
        reference_takes_list = []
        query_takes_list = []
        total_min_samples = 100
        for user_id in torch.unique(y):
            user_takes_with_counts = torch.tensor(list(zip(*torch.unique(take_ids[y == user_id], return_counts=True))))
            user_takes_with_counts = user_takes_with_counts[torch.randperm(user_takes_with_counts.size(0)), :]
            reference_takes_list.append(user_takes_with_counts[:n_reference_takes, 0].reshape(1, -1))
            query_takes_for_user = user_takes_with_counts[n_reference_takes:n_reference_takes + n_query_takes]
            min_samples = query_takes_for_user[:, 1].min()
            if min_samples < total_min_samples:
                total_min_samples = min_samples
            query_takes_list.append(query_takes_for_user[:, 0].reshape((1, -1, 2)))
        reference_takes = torch.cat(reference_takes_list, dim=0).to(embeddings.device)
        query_takes = torch.cat(query_takes_list, dim=0).to(embeddings.device)
        return query_takes, reference_takes

