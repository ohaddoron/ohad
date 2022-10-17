import torch
from torch import nn
import typing as tp

from torchtuples import tuplefy
from torchtuples.practical import DenseVanillaBlock
import torchtuples as tt


class DenseBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, activation_params=None,
                 dropout_rate: float = 0.):
        super(DenseBlock, self).__init__()
        if activation_params is None:
            activation_params = {}
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = getattr(nn, activation)(**activation_params)

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.activation(self.bn(self.linear(self.dropout(x))))


def convert_predictions_to_survival_prediction(surv_output: torch.Tensor):
    output = torch.ones_like(surv_output)
    cum_sum_surv_out = torch.cumsum(surv_output, dim=1)
    return output - cum_sum_surv_out


class SurvMLP(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_nodes: tp.Union[int, tp.List[int], tp.Tuple[int]],
                 survival_output_resolution: int = 100,
                 activation: str = 'ReLU',
                 dropout: float = 0.,
                 **kwargs
                 ):
        super(SurvMLP, self).__init__()

        if isinstance(hidden_nodes, int):
            hidden_nodes = [hidden_nodes]

        nodes = [in_features] + hidden_nodes

        self.layers = nn.ModuleList()

        for num_units_pre, num_units_post in zip(nodes[:-1], nodes[1:]):
            self.layers.append(DenseBlock(in_features=num_units_pre,
                                          out_features=num_units_post,
                                          activation=activation,
                                          dropout_rate=dropout
                                          )
                               )
        self.survival_layer = DenseBlock(in_features=nodes[-1],
                                         out_features=survival_output_resolution,
                                         activation='Softmax',
                                         dropout_rate=dropout)

    def forward(self, x):
        interm_output = [x]

        for layer in self.layers:
            interm_output.append(layer(interm_output[-1]))

        raw_surv_out = self.survival_layer(interm_output[-1])

        return convert_predictions_to_survival_prediction(raw_surv_out), interm_output


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
