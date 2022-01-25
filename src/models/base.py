from pydantic import BaseModel
from torch import nn, Tensor


class ZScoreLayer(nn.BatchNorm1d):
    def forward(self, input: Tensor, inverse: bool = False) -> Tensor:
        if not inverse:
            return super().forward(input)
        parameters = dict(list(self.named_parameters()))
        return (input * parameters['weight']) + parameters['bias']


class LayerDef(BaseModel):
    hidden_dim: int
    activation: str
    batch_norm: bool
