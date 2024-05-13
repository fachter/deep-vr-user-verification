from torch import nn

from src.models.positional_encodings import RelativePositionalEncoding, TimeAbsolutePositionalEncoding


class BaseTransformerModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        window_size: int,
        num_out_classes: int,
        d_model: int,
        dim_feedforward: int,
        pe_dropout: float,
        dropout_global: float,
        nhead: int,
        num_layers: int,
        positional_encoding: str,
        **_kwargs,
    ):
        super().__init__()

        self.hparams = dict(
            window_size=window_size,
            num_out_classes=num_out_classes,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            pe_dropout=pe_dropout,
            dropout_global=dropout_global,
            nhead=nhead,
            num_layers=num_layers,
            positional_encoding=positional_encoding,
        )
        
        self.num_out_classes = num_out_classes

        pe = str(self.hparams["positional_encoding"]).lower()
        if pe == "tape":
            self.pos_encoder = TimeAbsolutePositionalEncoding(
                d_model=d_model,
                dropout=pe_dropout,
                sequence_length=window_size,
            )
        elif pe == "rpe":
            self.pos_encoder = RelativePositionalEncoding(
                d_model=d_model,
                sequence_length=window_size,
            )
        elif pe in ["false", "missing", "none"]:
            self.pos_encoder = nn.Identity()
        else:
            raise Exception(f"Unknown positional encoding '{pe}'")

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                dim_feedforward=dim_feedforward,
                dropout=dropout_global,
            ),
            num_layers=num_layers,
        )

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)

        return x
