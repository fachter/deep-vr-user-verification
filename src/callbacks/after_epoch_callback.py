from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer


class AfterEpochCallback(Callback):
    def __init__(self, data_module):
        self.data_module = data_module

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pass