from torch import nn


class MLPModel(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_out_classes: int,
        layer_size: int,
        number_of_layers: int,
        activation_function: str,
    ):
        super().__init__()

        self.hparams = dict(
            layer_size=layer_size,
            number_of_layers=number_of_layers,
            activation_function=activation_function,
        )
        self.num_features = num_features
        self.num_out_classes = num_out_classes

        self.hidden_layers = self._build_hidden_layers()

        self.out = nn.Linear(self.hparams["layer_size"], self.num_out_classes)

    def _build_hidden_layers(self):
        hidden_layers = []
        in_channels = self.num_features
        for _layer_idx in range(self.hparams["number_of_layers"]):
            hidden_layers += [
                nn.Linear(in_channels, self.hparams["layer_size"]),
                getattr(nn, self.hparams["activation_function"])(),
            ]
            in_channels = self.hparams["layer_size"]
        return nn.Sequential(*hidden_layers)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.out(x)
        return x
