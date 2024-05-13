from torch import nn
from src.models.base_transformer_model import BaseTransformerModel


class SimpleTransformerModel(BaseTransformerModel):
    def __init__(
        self,
        num_features: int,
        window_size: int,
        num_out_classes: int,
        d_model: int,
        dim_feedforward: int,
        dropout_frames: float,
        pe_dropout: float,
        dropout_global: float,
        nhead: int,
        num_layers: int,
        positional_encoding: str,
        **_kwargs,
    ):
        super().__init__(
            num_features=num_features,
            window_size=window_size,
            num_out_classes=num_out_classes,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout_frames=dropout_frames,
            pe_dropout=pe_dropout,
            dropout_global=dropout_global,
            nhead=nhead,
            num_layers=num_layers,
            positional_encoding=positional_encoding,
        )

        self.frame_dropout = nn.Dropout(p=dropout_frames)

        self.projection_layer = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.Dropout(dropout_global),
            nn.Tanh(),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout_global),
            nn.ReLU(),
            nn.Linear(d_model, num_out_classes))

    def forward(self, x):
        x = self.frame_dropout(x)
        x = self.projection_layer(x)
        x = super().forward(x)
        x = self.output_layer(x)

        return x
