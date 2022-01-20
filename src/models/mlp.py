import torch
from pydantic import BaseModel
from torch import nn, jit, Tensor
import typing as tp

from src.models import LayerDef


class ZScoreLayer(nn.BatchNorm1d):
    def __init__(self, num_features, inverse: bool = False):
        super().__init__(num_features)
        self._inverse = inverse

    def forward(self, input: Tensor) -> Tensor:
        if not self._inverse:
            return super().forward(input)
        parameters = dict(list(self.named_parameters()))
        return (input * parameters['weight']) + parameters['bias']


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
        self._layers = layers

    def forward(self, x: torch.Tensor):

        for layer in self._layers:
            x = layer(x)

        return x
