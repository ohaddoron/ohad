import torch
from torch import nn
import typing as tp

from src.models import ZScoreLayer, LayerDef


class MLP(nn.Module):
    def __init__(self, input_features: int, layer_defs: tp.List[LayerDef], *args, **kwargs):

        super().__init__()

        layers = nn.ModuleList()

        last_layer_dim = input_features

        for layer_def in layer_defs:
            layers.append(nn.Linear(last_layer_dim, layer_def.hidden_dim))
            if layer_def.batch_norm:
                layers.append(nn.BatchNorm1d(layer_def.hidden_dim))
            layers.append(getattr(nn, layer_def.activation)())
            last_layer_dim = layer_def.hidden_dim
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
        self.zscore = ZScoreLayer(input_features)

        self.encoder = Encoder(input_features, encoder_layer_defs)
        self.decoder = Decoder(encoder_layer_defs[-1].hidden_dim, decoder_layer_defs)

    def forward(self, x, return_aux: bool = False):
        encoder_out = self.encoder(self.zscore(x))
        decoder_out = self.zscore.forward(self.decoder(encoder_out), inverse=False)
        if return_aux:
            return dict(out=decoder_out, aux=encoder_out)
        return decoder_out


class Attention1d(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=1)

    def forward(self, input):
        return self.conv(input).squeeze()


class AutoEncoderAttention(nn.Sequential):
    """Auto-encoder designed to encoder 1d data but can also handle attention to be provided to said 1d data. The
    attention in the main use case is the knowledge on which feature was knocked off so the network will have the
    apriori knowledge on which feature is it going to be tested on.

    :param input_channels: Number of input channels to be provided to the auto encoder. For a feature vector and accompanying attention, this should be set to 2
    :type input_channels: int
    :param input_features: Number of input features for the autoencoder to handle
    :type input_features: int
    :param encoder_layer_defs: Layer definitions for the encoder. See :class:`src.models.LayerDef`.
    :type encoder_layer_defs: List[:class:`src.models.LayerDef`]
    :param decoder_layer_defs: Layer definitions for the decoder. See :class:`src.models.LayerDef`
    :type decoder_layer_defs: List[:class:`src.models.LayerDef`]
    """

    def __init__(self, input_channels: int, input_features: int, encoder_layer_defs, decoder_layer_defs, *args):
        super().__init__(*args)
        self.autoencoder = AutoEncoder(input_features=input_features,
                                       encoder_layer_defs=encoder_layer_defs,
                                       decoder_layer_defs=decoder_layer_defs)
        self.attention = Attention1d(input_channels=input_channels, output_channels=1)

    def forward(self, input):
        return self.autoencoder(self.attention(input).squeeze())
