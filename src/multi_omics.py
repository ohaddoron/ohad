import os
import random
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import typer
from pydantic import BaseModel
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from typing import *

from common.database import init_database
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from src.models import LayerDef
from src.models.mlp import MultiHeadAutoEncoderRegressor
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import MultiOmicsDataset

app = typer.Typer()


def get_num_attributes(general_config, modality):
    attributes = init_database(general_config.DATABASE_CONFIG_NAME)[modality].distinct('name')
    input_features = len(attributes)
    return input_features


class GeneralConfig(BaseModel):
    modalities: List[str] = ['GeneExpression',
                             'CopyNumber'
                             ]

    DEBUG = getattr(sys, 'gettrace', None)() is not None
    DATABASE_CONFIG_NAME = 'brca-reader'
    OVERRIDE_ATTRIBUTES_FILE = True

    def __hash__(self):
        return hash(repr(self))


def get_patient_name_set(modalities: List[str], config_name: str) -> List[str]:
    db = init_database(config_name=config_name)
    patients = set()
    for modality in modalities:
        patients.update(set(db[modality].distinct('patient')))
    return list(patients)


@lru_cache
class DataConfig(BaseModel):
    """
    Data configuration
    """
    general_config = GeneralConfig()
    config_name = general_config.DATABASE_CONFIG_NAME
    batch_size = 4

    num_workers: int = 0 if general_config.DEBUG else os.cpu_count()

    patients = get_patient_name_set(general_config.modalities, config_name=general_config.DATABASE_CONFIG_NAME)
    train_patients = random.sample(patients, k=int(len(patients) * 0.9))
    val_patients = list(set(patients) - set(train_patients))
    collections = GeneralConfig().modalities

    def __hash__(self):
        return hash(repr(self))


@lru_cache
class TrainerConfig(BaseModel):
    """
    Trainer configuration
    """

    gpus: int = 1 if torch.cuda.is_available() else None
    auto_select_gpus = True
    # desired_batch_size = 16
    accumulate_grad_batches = max(1, 16 // DataConfig().batch_size)
    reload_dataloaders_every_epoch = False

    checkpoint_callback = True
    profiler = 'simple'
    fast_dev_run = GeneralConfig().DEBUG
    progress_bar_refresh_rate = 0
    max_epochs = int(1e6)

    default_root_dir = f'{tempfile.gettempdir()}/MultiOmics'
    stochastic_weight_avg = False


class MultiOmicsRegressorConfig(BaseModel):
    general_config = GeneralConfig()
    modalities = general_config.modalities

    modalities_model_def = {modality: dict(
        input_features=get_num_attributes(GeneralConfig(), modality=modality),
        encoder_layer_defs=[LayerDef(hidden_dim=1024, activation='Hardswish', batch_norm=True)],
        decoder_layer_defs=[
            LayerDef(hidden_dim=1024, activation='Mish', batch_norm=True),
            LayerDef(hidden_dim=(get_num_attributes(general_config=GeneralConfig(), modality=modality)),
                     activation='LeakyReLU', batch_norm=True)
        ],
        regressor_layer_defs=[
            LayerDef(hidden_dim=64, activation='Mish', batch_norm=True),
            LayerDef(hidden_dim=1, activation='ReLU', batch_norm=True)
        ]
    )
        for modality in modalities}

    lr = 1e-3

    def __hash__(self):
        return hash(repr(self))


class DataModule(LightningDataModule):
    def __init__(self,
                 train_patients: List[str],
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
                               )
        return DataLoader(dataset=ds,
                          batch_size=self._batch_size,
                          num_workers=self._num_workers,
                          shuffle=True,
                          drop_last=True
                          )


class MultiOmicsRegressor(LightningModule):
    def __init__(self,
                 modalities: List[str],
                 modalities_model_config: dict,
                 train_patients: List[str],
                 val_patients: List[str],
                 lr: float,
                 *args: Any,
                 **kwargs: Any):
        super().__init__()

        self.modalities = modalities

        self.models = torch.nn.ModuleDict(
            {
                modality: MultiHeadAutoEncoderRegressor(**model_config) for modality, model_config in
                modalities_model_config.items()
            }
        )
        self.lr = lr
        self._train_patients = train_patients
        self._val_patients = val_patients

        self.save_hyperparameters()

    def training_step(self, batch_idx, batch) -> STEP_OUTPUT:
        return self.step(batch, purpose='train')

    def validation_step(self, batch_idx, batch) -> Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='val')

    def test_step(self, batch_idx, batch) -> Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='test')

    def configure_optimizers(self):
        return [torch.optim.Adam(model.parameters(), lr=self.lr) for model in self.models.values()]

    def step(self, batch, purpose: str):
        return torch.tensor(0.)


@app.command()
def train(modalities: Optional[List[str]] = typer.Option(None)):
    general_config: GeneralConfig = GeneralConfig()
    data_config: DataConfig = DataConfig(general_config=general_config)
    trainer_config: TrainerConfig = TrainerConfig()
    multi_omics_regressor_config: MultiOmicsRegressorConfig = MultiOmicsRegressorConfig()
    os.makedirs(Path(trainer_config.default_root_dir, 'wandb').as_posix(), exist_ok=True)

    wandb_logger = WandbLogger(f"Attribute Filler {'-'.join([modality for modality in general_config.modalities])}",
                               log_model=True,
                               save_dir=trainer_config.default_root_dir)

    wandb_logger.experiment.config.update(dict(general_config=general_config.dict(),
                                               data_config=data_config.dict(),
                                               trainer_config=trainer_config.dict()))

    model = MultiOmicsRegressor(modalities_model_config=multi_omics_regressor_config.modalities_model_def,
                                train_patients=data_config.train_patients,
                                val_patients=data_config.val_patients,
                                **multi_omics_regressor_config.dict()
                                )

    trainer = Trainer(**trainer_config.dict(),
                      logger=[wandb_logger if not general_config.DEBUG else False],
                      callbacks=[LearningRateMonitor()]

                      )
    datamodule = DataModule(**data_config.dict())

    trainer.fit(model,
                datamodule=datamodule
                )


if __name__ == '__main__':
    app()
