import pytest
import torch
from torch.nn import MSELoss

from src.models.mlp import MLP, AutoEncoder
from src.models import LayerDef


class TestMLP:
    @pytest.mark.parametrize('layer_defs', [[LayerDef(hidden_dim=100,
                                                      activation='Mish',
                                                      batch_norm=True),
                                             LayerDef(hidden_dim=1000,
                                                      activation='LeakyReLU',
                                                      batch_norm=True)
                                             ],
                                            [LayerDef(hidden_dim=100,
                                                      activation='Hardswish',
                                                      batch_norm=True),
                                             LayerDef(hidden_dim=1000,
                                                      activation='LeakyReLU',
                                                      batch_norm=True)
                                             ]
                                            ])
    def test_forward(self, layer_defs):
        model = MLP(input_features=1000, layer_defs=layer_defs)
        out = model(torch.rand(4, 1000))
        loss = MSELoss()(torch.zeros_like(out), out)
        loss.backward()
        assert any(list(model.layers[0].parameters())[0].grad)
        assert any(list(model.layers[0].parameters())[1].grad)


class TestAutoEncoder:
    def test_auto_encoder(self):
        model = AutoEncoder(input_features=1000,
                            encoder_layer_defs=[LayerDef(hidden_dim=100,
                                                         activation='Mish',
                                                         batch_norm=True),
                                                LayerDef(hidden_dim=10,
                                                         activation='Mish',
                                                         batch_norm=True)
                                                ],
                            decoder_layer_defs=[LayerDef(hidden_dim=10,
                                                         activation='Hardswish',
                                                         batch_norm=True),
                                                LayerDef(hidden_dim=100,
                                                         activation='Hardswish',
                                                         batch_norm=True)
                                                ]
                            )
        out = model(torch.rand(4, 1000), return_aux=True)
        assert isinstance(out, dict)
        assert {'out', 'aux'} == set(out.keys())
        loss = MSELoss()(torch.zeros_like(out['out']), out['out'])
        loss.backward()
