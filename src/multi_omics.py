import os
import random
import sys
import tempfile
import warnings
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
from torch.nn import TripletMarginLoss, MSELoss
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler, SubsetRandomSampler

from src.dataset import MultiOmicsDataset

app = typer.Typer()

warnings.filterwarnings("ignore")


def get_num_attributes(general_config, modality):
    attributes = init_database(general_config.DATABASE_CONFIG_NAME)[modality].distinct('name')
    input_features = len(attributes)
    return input_features


class GeneralConfig(BaseModel):
    modalities: List[str] = [
        'GeneExpressionGDC',
        'CopyNumber',
        'PathwayActivity',
        'TranscriptionFactorHiSeqV2'

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
    progress_bar_refresh_rate = 5
    max_epochs = int(1e6)

    default_root_dir = f'{tempfile.gettempdir()}/MultiOmics'
    stochastic_weight_avg = False
    limit_train_batches = 0.01


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
    loss_config = dict(
        triplet_loss=dict(margin=1.0, p=2.0, eps=1e-06, swap=True),
        autoencoding_loss=dict()
    )
    loss_weight_dict = dict(triplet_loss=0.3, autoencoding_loss=0.7)

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
                          num_workers=self._num_workers,
                          batch_size=None,
                          sampler=BatchSampler(sampler=RandomSampler(data_source=ds), batch_size=self._batch_size,
                                               drop_last=True),
                          collate_fn=None
                          )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        ds = MultiOmicsDataset(patients=self._val_patients,
                               collections=self._collections,
                               )
        return DataLoader(dataset=ds,
                          num_workers=self._num_workers,
                          batch_size=None,
                          sampler=BatchSampler(sampler=SequentialSampler(data_source=ds), batch_size=self._batch_size,
                                               drop_last=True),
                          collate_fn=None
                          )


class MultiOmicsRegressor(LightningModule):
    def __init__(self,
                 modalities: List[str],
                 modalities_model_config: dict,
                 train_patients: List[str],
                 val_patients: List[str],
                 lr: float,
                 loss_config: dict,
                 loss_weight_dict: dict,
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
        self.loss_config = loss_config
        self.loss_weight_dict = loss_weight_dict

        self.losses = self.losses_definitions()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.step(batch, purpose='train')

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='val')

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.models.parameters(), lr=self.lr)

    def step(self, batch, purpose: str):
        anchor_out = self.models[batch['anchor_modality']](batch['anchor'])
        pos_out = self.models[batch['pos_modality']](batch['pos'])
        neg_out = self.models[batch['neg_modality']](batch['neg'])

        triplet_loss = self.losses['triplet_loss']['fn'](
            anchor_out['encoder'],
            pos_out['encoder'],
            neg_out['encoder']
        )

        self.log(name=f'{purpose}/triplet_loss', value=triplet_loss, on_step=True, on_epoch=True, sync_dist=True)

        anchor_reg = self.losses['autoencoding_loss']['fn'](anchor_out['autoencoder'], batch['anchor'])
        self.log(f'{purpose}/{batch["anchor_modality"]}_reg', anchor_reg, on_step=True, on_epoch=True, sync_dist=True)
        pos_reg = self.losses['autoencoding_loss']['fn'](pos_out['autoencoder'], batch['pos'])
        self.log(f'{purpose}/{batch["pos_modality"]}_reg', pos_reg, on_step=True, on_epoch=True, sync_dist=True)
        neg_reg = self.losses['autoencoding_loss']['fn'](neg_out['autoencoder'], batch['neg'])
        self.log(f'{purpose}/{batch["neg_modality"]}_reg', neg_reg, on_step=True, on_epoch=True, sync_dist=True)

        regression_loss = sum((anchor_reg, pos_reg, neg_reg)) / 3
        self.log('regression_loss', value=regression_loss, on_step=True, on_epoch=True, sync_dist=True)

        return self.losses['triplet_loss']['w'] * triplet_loss + self.losses[
            'autoencoding_loss']['w'] * regression_loss

    def losses_definitions(self):
        return dict(
            triplet_loss=dict(
                w=self.loss_weight_dict['triplet_loss'],
                fn=TripletMarginLoss(**self.loss_config['triplet_loss'])
            ),
            autoencoding_loss=dict(
                w=self.loss_weight_dict['autoencoding_loss'],
                fn=MSELoss(**self.loss_config['autoencoding_loss'])
            )
        )


@app.command()
def train(modalities: Optional[List[str]] = typer.Option(None)):
    general_config: GeneralConfig = GeneralConfig()
    data_config: DataConfig = DataConfig(general_config=general_config)
    trainer_config: TrainerConfig = TrainerConfig()
    multi_omics_regressor_config: MultiOmicsRegressorConfig = MultiOmicsRegressorConfig()
    os.makedirs(Path(trainer_config.default_root_dir, 'wandb').as_posix(), exist_ok=True)

    wandb_logger = WandbLogger(f"MultiOmics",
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
                      callbacks=[LearningRateMonitor()],
                      auto_select_gpus=True

                      )
    datamodule = DataModule(**data_config.dict())

    trainer.fit(model,
                datamodule=datamodule,
                )


if __name__ == '__main__':
    app()
