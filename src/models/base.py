from pydantic import BaseModel
from torch import nn, Tensor
import typing as tp


class ZScoreLayer(nn.BatchNorm1d):
    def forward(self, input: Tensor, inverse: bool = False) -> Tensor:
        if not inverse:
            return super().forward(input)
        parameters = dict(list(self.named_parameters()))
        return (input * parameters['weight']) + parameters['bias']


class LayerDef(BaseModel):
    hidden_dim: tp.Optional[int]
    activation: tp.Optional[str]
    batch_norm: tp.Optional[bool]
    layer_type: str = 'Linear'
    params: dict = None
