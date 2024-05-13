import os
from torch.utils.data import DataLoader
from src.datamodules.base_datamodule import BaseDatamodule


NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 8))


class EmbeddingDatamodule(BaseDatamodule):
    def train_dataloader(self):
        settings = self.data_loader_settings.copy()
        return DataLoader(self.train_dataset, **settings, shuffle=True)

    def val_dataloader(self):
        settings = self.data_loader_settings.copy()
        self.val_dataset.evaluation_mode = True
        return DataLoader(self.val_dataset, **settings)

    def test_dataloader(self):
        settings = self.data_loader_settings.copy()
        self.test_dataset.evaluation_mode = True
        return DataLoader(self.test_dataset, **settings)
