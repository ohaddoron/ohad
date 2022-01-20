from pydantic import BaseModel
from torch import nn, Tensor


class ZScoreLayer(nn.BatchNorm1d):
    def __init__(self, num_features, inverse: bool = False):
        super().__init__(num_features)
        self._inverse = inverse

    def forward(self, input: Tensor) -> Tensor:
        if not self._inverse:
            return super().forward(input)
        parameters = dict(list(self.named_parameters()))
        return (input * parameters['weight']) + parameters['bias']


class LayerDef(BaseModel):
    hidden_dim: int
    activation: str
    batch_norm: bool
