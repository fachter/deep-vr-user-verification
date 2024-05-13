import time
from typing import Any, List, Dict

import lightning as L
import numpy as np
import pytorch_metric_learning.losses
import torch
import wandb
from sklearn.calibration import LabelEncoder
from torch import nn

from src.custom_metrics.accuracy_calculator import MotionAccuracyCalculator
from src.utils import utils

logger = utils.get_logger(__name__)


class EmbeddingModule(L.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            loss_weights=None,
            optimizer_options: dict = None,
            metrics: List[dict] = None,
            klasses: List[str] = None,
            loss_options: Dict[str, Any] = None,
    ):
        super().__init__()
        self.best_logged_metrics = {}
        self.klasses = klasses
        self.num_classes = len(klasses)
        self.optimizer_options = {} if optimizer_options is None else optimizer_options
        self.hyperparameters = model.hparams
        self.model = model
        self.evaluator = MotionAccuracyCalculator(
            sequence_lengths_minutes=[1, 2, 5],
            sliding_window_step_size_seconds=1,
            k="max_bin_count",
            device=torch.device("cpu")
        )

        if wandb.run is not None:
            wandb.define_metric("precision_at_1/validation", summary="max")
            wandb.define_metric("per_take_accuracy/validation", summary="max")
            wandb.define_metric("per_user_accuracy/validation", summary="max")
            wandb.define_metric("loss_train", summary="min")

        self.batch_tuning_mode = False
        self.loss_func = self._initialize_loss_function(loss_options)
        # self.validation_step_outputs = []
        self.validation_step_predictions = []
        self.validation_step_targets = []
        self.validation_step_take_ids = []
        self.validation_step_session_indices = []

        if loss_weights is not None:
            self.loss_weights = torch.from_numpy(loss_weights).float().to(self.device)
        else:
            self.loss_weights = None
        self.save_hyperparameters()

    def _initialize_loss_function(self, loss_options):
        loss_options = loss_options.copy()
        loss_options: Dict[str, Any] = (
            {
                "name": "NormalizedSoftmaxLoss",
                "num_classes": "auto",
                "embedding_size": "auto",
            }
            if loss_options is None
            else loss_options
        )

        if loss_options.get("num_classes") == "auto":
            loss_options["num_classes"] = self.num_classes
        if loss_options.get("embedding_size") == "auto":
            loss_options["embedding_size"] = self.model.num_out_classes

        loss_func = getattr(pytorch_metric_learning.losses, loss_options.pop("name"))(
            **loss_options
        )
        return loss_func

    def forward(self, X):
        embeddings = self.model.forward(X.float())
        return embeddings

    def training_step(self, batch, batch_idx):
        X = batch["data"]
        y = batch["targets"]

        embedding = self.forward(X)
        loss = self.loss_func(embedding, y.long())
        self.log(
            f"loss_train", loss, on_step=True, on_epoch=True
        )

        return loss

    def configure_optimizers(self):
        optimizer_name = self.optimizer_options.pop("name", "Adam")
        optimizer = getattr(torch.optim, optimizer_name)(
            params=self.model.parameters(), **self.optimizer_options
        )

        return optimizer

    def validation_step(self, batch, _batch_idx):
        X = batch["data"]
        y = batch["targets"]
        # take_ids = batch["take_id"]
        session_index = batch['session_idx']

        h = self.forward(X)
        # self.validation_step_predictions.append(h)
        self.validation_step_predictions.append(h)
        self.validation_step_targets.append(y)
        self.validation_step_session_indices.append(session_index)
        # self.validation_step_take_ids.append(take_ids)
        # self.validation_step_outputs.append((h, y, take_ids))

    def predict_step(self, batch, batch_idx, **kwargs):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        if self.batch_tuning_mode:
            return

        embeddings = torch.cat(self.validation_step_predictions).cpu()
        y = torch.cat(self.validation_step_targets).cpu()
        session_indices = torch.cat(self.validation_step_session_indices).cpu()
        # take_ids = torch.Tensor(LabelEncoder().fit_transform(torch.cat(self.validation_step_take_ids).cpu()))
        # embeddings = torch.cat([emb for emb, _, _ in self.validation_step_outputs]).cpu()
        # y = torch.cat([y for _, y, _ in self.validation_step_outputs]).cpu()
        # take_ids = torch.Tensor(
        #     LabelEncoder().fit_transform(
        #         torch.Tensor([tid for _, _, tids in self.validation_step_outputs for tid in tids])))

        # self._compute_and_log_validation_metrics_with_takes(embeddings, y, take_ids)
        self._compute_and_log_validation_metrics(embeddings, y, session_indices)
        self._note_best_metric_values()

        # self.validation_step_outputs = []
        self.validation_step_predictions = []
        self.validation_step_targets = []
        self.validation_step_take_ids = []
        self.validation_step_session_indices = []

    def _compute_and_log_validation_metrics(self, embeddings, y, session_idxs):
        # speeds up cpu calculations
        torch.set_num_threads(8)

        session_1_embeddings = embeddings[session_idxs == 0].cpu()
        session_1_y = y[session_idxs == 0].cpu()
        session_2_embeddings = embeddings[session_idxs == 1].cpu()
        session_2_y = y[session_idxs == 1].cpu()

        reference_embeddings = session_1_embeddings[::150].contiguous()
        reference_y = session_1_y[::150].contiguous()
        index_to_skip = torch.randint(0, 3, (1, )).item()
        query_mask = torch.arange(session_2_embeddings.size(0)) % 4 != index_to_skip
        query_embeddings = session_2_embeddings[query_mask].contiguous()
        query_y = session_2_y[query_mask].contiguous()

        if query_embeddings.any() and reference_embeddings.any():
            with torch.no_grad():
                with torch.cuda.device(-1):
                    with torch.autocast(enabled=False, device_type="cpu"):
                        accuracy = self.evaluator.get_accuracy(query_embeddings, query_y, reference_embeddings,
                                                               reference_y,
                                                               ref_includes_query=False)
            self.log_metrics(accuracy, "validation")

    def _compute_and_log_validation_metrics_with_takes(self, embeddings, y, take_ids: torch.Tensor):
        torch.set_num_threads(8)

        n_reference_takes = 1
        num_users = len(torch.unique(y))

        reference_takes_list = []
        query_takes_list = []

        for user_id in torch.unique(y):
            user_takes = torch.unique(take_ids[y == user_id])
            reference_takes_list.append(user_takes[:n_reference_takes])
            query_takes_list.append(user_takes[n_reference_takes:])

        reference_takes = torch.cat(reference_takes_list)

        assert len(reference_takes) == num_users * n_reference_takes, (f"{len(reference_takes)} != "
                                                                       f"{num_users} * {n_reference_takes}")

        reference_mask = torch.isin(take_ids, reference_takes)
        query_mask = ~reference_mask

        reference_embeddings = embeddings[reference_mask]
        reference_y = y[reference_mask]
        query_embeddings = embeddings[query_mask]
        query_y = y[query_mask]

        if query_embeddings.any() and reference_embeddings.any():
            try:
                start = time.time()
                with torch.no_grad():
                    with torch.cuda.device(-1):
                        with torch.autocast(enabled=False, device_type="cpu"):
                            accuracy = self.evaluator.get_accuracy(
                                query_embeddings,
                                query_y,
                                reference_embeddings,
                                reference_y,
                                query_take_ids=take_ids[query_mask],
                                ref_includes_query=False,
                            )
                end = time.time()
                self.log_metrics(accuracy, "validation")
                self.log("validation_computation_time", end - start)
            except RuntimeError as e:
                logger.warning(f"error while computing validation metrics: {e}")

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
