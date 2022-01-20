import torch
from torch import nn
import typing as tp

from src.models import ZScoreLayer, LayerDef


class MLP(nn.Module):
    def __init__(self, input_features: int, layer_defs: tp.List[LayerDef], *args, **kwargs):

        super().__init__()

        layers = nn.ModuleList()
        layers.append(ZScoreLayer(input_features))

        last_layer_dim = input_features

        for layer_def in layer_defs:
            layers.append(nn.Linear(last_layer_dim, layer_def.hidden_dim))
            if layer_def.batch_norm:
                layers.append(nn.BatchNorm1d(layer_def.hidden_dim))
            layers.append(getattr(nn, layer_def.activation)())
            last_layer_dim = layer_def.hidden_dim
        layers.append(ZScoreLayer(last_layer_dim, inverse=True))
        self.layers = layers

    def forward(self, x: torch.Tensor):

        for layer in self.layers:
            x = layer(x)
        return x


class Encoder(MLP):
    def __init__(self, input_features: int, layer_defs: tp.List[LayerDef], *args, **kwargs):
        super().__init__(input_features, layer_defs, *args, **kwargs)
        if isinstance(self.layers[-1], ZScoreLayer):
            self.layers = self.layers[:-1]


class Decoder(MLP):
    def __init__(self, input_features: int, layer_defs: tp.List[LayerDef], *args, **kwargs):
        super().__init__(input_features, layer_defs, *args, **kwargs)
        if isinstance(self.layers[0], ZScoreLayer):
            self.layers = self.layers[1:]


class AutoEncoder(nn.Module):
    def __init__(self, input_features, encoder_layer_defs: tp.List[LayerDef], decoder_layer_defs: tp.List[LayerDef]):
        super().__init__()
        self.encoder = Encoder(input_features, encoder_layer_defs)
        self.decoder = Decoder(encoder_layer_defs[-1].hidden_dim, decoder_layer_defs)

    def forward(self, x, return_aux: bool = False):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        if return_aux:
            return dict(out=decoder_out, aux=encoder_out)
        return decoder_out
