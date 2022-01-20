from pydantic import BaseModel


class LayerDef(BaseModel):
    hidden_dim: int
    activation: str
    batch_norm: bool
