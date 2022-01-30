import os
import sys
from dataclasses import dataclass

import typer
from pydantic import BaseModel
from pytorch_lightning import LightningModule, LightningDataModule
from typing import *

from common.database import init_database
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from src.models import LayerDef
from src.models.mlp import MultiHeadAutoEncoderRegressor
from torch.utils.data import DataLoader

from ohad.src.dataset import MultiOmicsDataset

app = typer.Typer()


class GeneralConfig(BaseModel):
    modalities: List[str] = ['GeneExpression',
                             'CopyNumber'
                             ]

    DEBUG = getattr(sys, 'gettrace', None)() is not None
    DATABASE_CONFIG_NAME = 'omicsdb'
    OVERRIDE_ATTRIBUTES_FILE = True


def get_num_attributes(general_config, modality):
    attributes = init_database(general_config.DATABASE_CONFIG_NAME)[modality].distinct('name')
    input_features = len(attributes)
    return input_features


class MultiOmicsRegressorConfig(BaseModel):
    general_config = GeneralConfig()
    modalities = general_config.modalities

    modalities_model_def = {modality: dict(
        encoder_layers_def=[LayerDef(hidden_dim=1024, activation='Hardswish', batch_norm=True)],
        decoder_layers_def=[
            LayerDef(hidden_dim=1024, activation='Mish', batch_norm=True),
            LayerDef(hidden_dim=(get_num_attributes(general_config=GeneralConfig(), modality=modality)),
                     activation='LeakyReLU', batch_norm=True)
        ],
        regressor_layers_def=[
            LayerDef(hidden_dim=64, activation='Mish', batch_norm=True),
            LayerDef(hidden_dim=1, activation='ReLU', batch_norm=True)
        ]
    )
        for modality in modalities}

    pass


class DataModule(LightningDataModule):
    def __init__(self, train_patients: List[str],
                 val_patients: List[str],
                 collections: List[str],
                 batch_size: int,
                 config_name: str,
                 num_workers=None, *args, **kwargs):
        super().__init__()

        assert not set(train_patients).intersection(set(val_patients))

        self._train_patients = train_patients
        self._val_patients = val_patients
        self._collections = collections

        self._batch_size = batch_size
        self._num_workers = num_workers or max(os.cpu_count(), 10)
        self._config_name = config_name

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        ds = MultiOmicsDataset(patients=self._train_patients,
                               collections=self._collections,
                               config_name=self._config_name,

                               )
        return DataLoader(dataset=ds,
                          batch_size=self._batch_size,
                          num_workers=self._num_workers,
                          shuffle=True,
                          drop_last=True
                          )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        ds = MultiOmicsDataset(patients=self._val_patients,
                               collections=self._collections,
                               config_name=self._config_name,

                               )
        return DataLoader(dataset=ds,
                          batch_size=self._batch_size,
                          num_workers=self._num_workers,
                          shuffle=True,
                          drop_last=True
                          )


class MultiOmicsRegressor(LightningModule):
    def __init__(self, modalities: List[str], modalities_model_config: dict, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.modalities = modalities

        self.models = {
            modality: MultiHeadAutoEncoderRegressor(**model_config) for modality, model_config in
            modalities_model_config.items()
        }

    def step(self, batch):
        pass


if __name__ == '__main__':
    app()
