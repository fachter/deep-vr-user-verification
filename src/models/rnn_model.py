from torch import nn


class RNNModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_out_classes: int,
        num_rnn_layers: int,
        rnn_hidden_size: int,
        dropout: float,
        cell_type: str,
        **_kwargs
    ):
        super().__init__()

        self.num_features = num_features
        self.num_out_classes = num_out_classes
        self.hparams = dict(
            num_rnn_layers=num_rnn_layers,
            rnn_hidden_size=rnn_hidden_size,
            dropout=dropout,
            cell_type=cell_type,
        )
        self.rnn = getattr(nn, self.hparams["cell_type"])(
            input_size=self.num_features,
            hidden_size=self.hparams["rnn_hidden_size"],
            batch_first=True,
            dropout=self.hparams["dropout"],
            num_layers=self.hparams["num_rnn_layers"],
        )
        self.output_layer = nn.Linear(
            self.hparams["rnn_hidden_size"], self.num_out_classes
        )

    def forward(self, x):
        x, _rnn_state = self.rnn(x)
        x = x[:, -1]
        x = self.output_layer(x)
        return x
