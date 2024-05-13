import wandb
from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer


class LogBestMetrics(Callback):
    def on_validation_end(self, trainer: Trainer, lightning_module: LightningModule):
        self._add_best_logged_metrics_to_wandb(trainer, lightning_module)

    def _add_best_logged_metrics_to_wandb(self, trainer: Trainer, identifier: LightningModule):
        for (metric_name, direction), best_value in identifier.best_logged_metrics.items():
            target_direction = "min" if "loss" in metric_name else "max"

            if target_direction == direction:
                wandb.run.summary[f"best_{metric_name}"] = best_value
