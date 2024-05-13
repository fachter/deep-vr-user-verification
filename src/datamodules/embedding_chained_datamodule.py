import os
from torch.utils.data import DataLoader
from src.datamodules.base_chained_datamodule import BaseChainedDatamodule


NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 2))
# NUM_WORKERS = 0


class EmbeddingChainedDatamodule(BaseChainedDatamodule):
    def train_dataloader(self):
        settings = self.data_loader_settings.copy()
        return DataLoader(self.train_dataset, **settings)

    def val_dataloader(self):
        settings = self.data_loader_settings.copy()
        self.val_dataset.evaluation_mode = True
        return DataLoader(self.val_dataset, **settings)

    def test_dataloader(self):
        settings = self.data_loader_settings.copy()
        self.test_dataset.evaluation_mode = True
        return DataLoader(self.test_dataset, **settings)
