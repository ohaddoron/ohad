import torch
from torch import nn
import typing as tp

from torchtuples import tuplefy
from torchtuples.practical import DenseVanillaBlock
import torchtuples as tt


class MLPVanilla(nn.Module):
    """A version of torchtuples.practical.MLPVanilla that works for CoxTime.
    The difference is that it takes `time` as an additional input and removes the output bias and
    output activation.
    """

    def __init__(self, in_features, num_nodes, batch_norm=True, dropout=None, activation: str = 'ReLU',
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        in_features += 1
        out_features = 1
        output_activation = None
        output_bias = False
        self.last_layer_bn = nn.BatchNorm1d(1)
        self.last_layer_relu = nn.ReLU()
        self.net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                           getattr(nn, activation), output_activation, output_bias, w_init_)

    def forward(self, input, time):
        input = torch.cat([input, time], dim=1)
        return self.net(input)
