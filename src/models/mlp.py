import torch
from pydantic import BaseModel
from torch import nn, jit
import typing as tp

from src.models import LayerDef


class ZScoreLayer(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.input_bias = nn.Parameter(torch.zeros((input_features,)), requires_grad=True)
        self.input_variance = nn.Parameter(torch.ones((input_features,)), requires_grad=True)

    def forward(self, x):
        return (x - self.input_bias) / self.input_variance

    @jit.export
    def inverse(self, x):
        return (x * self.input_variance) + self.input_bias


class MLP(nn.Module):
    def __init__(self, input_features: int, layer_defs: tp.List[LayerDef]):

        super().__init__()

        self.zscore_layer: ZScoreLayer = ZScoreLayer(input_features=input_features)

        layers = nn.ModuleList()

        last_layer_dim = input_features

        for layer_def in layer_defs:
            layers.append(nn.Linear(last_layer_dim, layer_def.hidden_dim))
            if layer_def.batch_norm:
                layers.append(nn.BatchNorm1d(layer_def.hidden_dim))
            layers.append(getattr(nn, layer_def.activation)())
            last_layer_dim = layer_def.hidden_dim

        self._layers = layers

    def forward(self, x: torch.Tensor):
        x = self.zscore_layer(x)

        for layer in self._layers:
            x = layer(x)

        return self.zscore_layer.inverse(x)
