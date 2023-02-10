import torch
from torch import nn
import typing as tp

from torch.nn import Softmax
from torchtuples import tuplefy
from torchtuples.practical import DenseVanillaBlock
import torchtuples as tt


class DenseBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation: tp.Union[str, tp.Callable],
                 activation_params=None,
                 dropout_rate: float = 0.):
        super(DenseBlock, self).__init__()
        if activation_params is None:
            activation_params = {}
        self.bn = nn.BatchNorm1d(out_features)
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)(**activation_params)
        else:
            self.activation = activation

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.activation(self.bn(self.linear(self.dropout(x))))


class DeepSetsPhiMLP(nn.Module):
    """
    Phi projection based on deep sets paper for emebddings
    """

    def __init__(self,
                 hidden_nodes: tp.List[int],
                 in_features: int = 32,
                 out_features: int = 32,
                 activations: str = 'ReLU'
                 ):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            *[DenseBlock(in_features=in_features, out_features=hidden_node, activation=activations)
              for hidden_node in
              hidden_nodes]
        )
        self.out_layer = DenseBlock(
            out_features=out_features, activation=activations, in_features=in_features)

    def forward(self, x):
        return self.out_layer(self.hidden_layers(x))


class DeepSetsPhiTransformer(nn.TransformerEncoderLayer):
    pass


def convert_predictions_to_survival_prediction(surv_output: torch.Tensor):
    output = torch.ones_like(surv_output)
    cum_sum_surv_out = torch.cumsum(surv_output, dim=1)
    return torch.clamp(output - cum_sum_surv_out, min=0, max=1)


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
        self.survival_layer = DenseBlock(in_features=nodes[-2],
                                         out_features=survival_output_resolution,
                                         activation='Softmax',
                                         dropout_rate=dropout)

    def forward(self, x):
        interm_output = [x]

        for layer in self.layers:
            interm_output.append(layer(interm_output[-1]))

        raw_surv_out = self.survival_layer(interm_output[-2])

        return convert_predictions_to_survival_prediction(raw_surv_out), interm_output


class SurvAE(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_nodes: tp.Union[int, tp.List[int], tp.Tuple[int]],
                 survival_output_resolution: int = 100,
                 activation: str = 'ReLU',
                 dropout: float = 0.,
                 **kwargs
                 ):
        super(SurvAE, self).__init__()

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
        self.layers.append(DenseBlock(in_features=hidden_nodes[-1],
                                      out_features=in_features,
                                      activation=nn.Identity(),
                                      dropout_rate=dropout
                                      )
                           )
        self.survival_layer = DenseBlock(in_features=hidden_nodes[-1],
                                         out_features=survival_output_resolution,
                                         activation=Softmax(dim=1),
                                         dropout_rate=dropout)

    def forward(self, x):
        interm_output = [x]

        for layer in self.layers:
            interm_output.append(layer(interm_output[-1]))

        raw_surv_out = self.survival_layer(interm_output[-2])

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


class CnvNet(nn.Module):
    def __init__(self, in_features, net_params: dict, embedding_dims=(3, 2), **kwargs):
        super().__init__()
        embedding_dims = [embedding_dims] * in_features
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in embedding_dims])

        n_embeddings = in_features * 2

        self.fc = self._init_net(in_features=n_embeddings, **net_params)

    @staticmethod
    def _init_net(name: str, **kwargs) -> nn.Module:
        return globals()[name](**kwargs)

    def forward(self, x):
        x = x.to(torch.int64)

        x = [emb_layer(x[:, i] + 1)
             for i, emb_layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        out = self.fc(x)

        return out


class ClinicalNet(nn.Module):
    """Clinical data extractor.
    Handle continuous features and categorical feature embeddings.
    """

    def __init__(self, net_params, embedding_dims=None, **kwargs):
        super(ClinicalNet, self).__init__()
        # Embedding layer
        if embedding_dims is None:
            embedding_dims = [
                (33, 17), (2, 1), (8, 4), (3, 2), (3, 2), (3, 2), (3, 2), (3, 2),
                (20, 10)]
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y)
                                               for x, y in embedding_dims])

        n_embeddings = sum([y for x, y in embedding_dims])
        n_continuous = 1

        # Linear Layers
        self.linear = nn.Linear(n_embeddings + n_continuous, 256)

        # Embedding dropout Layer
        self.embedding_dropout = nn.Dropout()

        # Continuous feature batch norm layer
        self.bn_layer = nn.BatchNorm1d(n_continuous)

        # Output Layer
        self.output_layer = self._init_net(in_features=256, **net_params)

    @staticmethod
    def _init_net(name: str, **kwargs) -> nn.Module:
        return globals()[name](**kwargs)

    def forward(self, x):
        continuous_x, categorical_x = x[:, :1], x[:, 1:]
        categorical_x = categorical_x.to(torch.int64)

        x = [emb_layer(categorical_x[:, i])
             for i, emb_layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, 1)
        x = self.embedding_dropout(x)

        continuous_x = self.bn_layer(continuous_x)

        x = torch.cat([x, continuous_x], 1)
        out = self.output_layer(self.linear(x))

        return out


class ClinicalNetAttention(nn.Module):
    def __init__(self, net_params, embedding_dims=None, **kwargs):
        super().__init__()
        if embedding_dims is None:
            embedding_dims = [
                (33, 32), (2, 32), (8, 32), (3, 32), (3,
                                                      32), (3, 32), (3, 32), (3, 32),
                (20, 32)]
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y)
                                               for x, y in embedding_dims])

        n_embeddings = sum([y for x, y in embedding_dims])
        n_continuous = 1

        # Linear Layers
        self.continuous_linear = nn.Linear(1, 32)

        self.attention = nn.TransformerEncoderLayer(d_model=32, nhead=8)

        self.linear = nn.Linear(10, 256)

        # Embedding dropout Layer
        self.embedding_dropout = nn.Dropout()

        # Continuous feature batch norm layer
        self.bn_layer = nn.BatchNorm1d(n_continuous)

        # Output Layer
        self.output_layer = self._init_net(in_features=256, **net_params)

    @staticmethod
    def _init_net(name: str, **kwargs) -> nn.Module:
        return globals()[name](**kwargs)

    def forward(self, x):
        continuous_x, categorical_x = x[:, :1], x[:, 1:]
        categorical_x = categorical_x.to(torch.int64)

        x = [emb_layer(categorical_x[:, i])
             for i, emb_layer in enumerate(self.embedding_layers)]

        continuous = self.continuous_linear(continuous_x).unsqueeze(1)
        x = torch.stack(x, 1)

        x = torch.cat([x, continuous], axis=1)

        x = self.attention(x)

        x = torch.mean(x, dim=-1)

        out = self.output_layer(self.linear(x))

        return out
