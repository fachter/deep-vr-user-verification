import torch
from torch import nn

from src.models._cnn_layer import CNNLayer
from src.utils.normalization_helper import normalize_embedding


class CNNModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        window_size: int,
        num_out_classes: int,
        num_layers: int,
        kernel_size: int,
        conv_stride: int,
        initial_channel_size: int,
        channels_factor: int,
        dropout: float,
        max_pool_size: int,
        activation: str = "ReLU",
        normalize_model_outputs: bool = False,
        **_kwargs
    ):
        super().__init__()
        self.num_features = num_features
        self.num_out_classes = num_out_classes
        
        self.hparams = dict(
            num_layers=num_layers,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            initial_channel_size=initial_channel_size,
            channels_factor=channels_factor,
            dropout=dropout,
            max_pool_size=max_pool_size,
            activation=activation,
            normalize_model_outputs=normalize_model_outputs,
        )

        self.ops = nn.Sequential(
            *[
                CNNLayer(
                    kernel_size=self.hparams["kernel_size"],
                    out_channels=int(
                        self.hparams["initial_channel_size"]
                        * self.hparams["channels_factor"] ** layer_idx
                    ),
                    conv_stride=self.hparams["conv_stride"],
                    max_pool_kernel_size=self.hparams["max_pool_size"],
                    dropout=self.hparams["dropout"],
                    activation=self.hparams["activation"],
                )
                for layer_idx in range(self.hparams["num_layers"])
            ],
            nn.Flatten(),
            nn.LazyLinear(num_out_classes)
        )

        # required to init the lazy modules,
        #               otherwise PyTorch Lightning will raise an error during startup
        self.ops(torch.zeros((1, num_features, window_size)))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.ops(x)

        if self.hparams["normalize_model_outputs"]:
            x = normalize_embedding(x)

        return x
