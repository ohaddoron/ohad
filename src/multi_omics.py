import json
import os
import random
import sys
import tempfile
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from torch.nn.functional import mse_loss
from torchmetrics import SymmetricMeanAbsolutePercentageError
from torchmetrics.functional import precision_recall
import wandb
from src.logger import logger
import toml
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
from torch.nn import TripletMarginLoss, MSELoss, TripletMarginWithDistanceLoss
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
    batch_size = 32

    num_workers: int = 0 if general_config.DEBUG else min(os.cpu_count(), 16)

    patients = get_patient_name_set(general_config.modalities, config_name=general_config.DATABASE_CONFIG_NAME)
    train_patients = random.sample(patients, k=int(len(patients) * 0.9))
    val_patients = list(set(patients) - set(train_patients))
    collections = general_config.modalities
    short_long_survival_cutoff = 921.  # This value is set based on the median survival times

    def __hash__(self):
        return hash(repr(self))


@lru_cache
class TrainerConfig(BaseModel):
    """
    Trainer configuration
    """

    gpus: int = [1] if torch.cuda.is_available() else None
    auto_select_gpus = False
    # desired_batch_size = 32
    accumulate_grad_batches = max(1, 32 // DataConfig().batch_size)
    reload_dataloaders_every_epoch = False

    enable_checkpointing = True

    profiler = 'simple'
    fast_dev_run = GeneralConfig().DEBUG
    progress_bar_refresh_rate = 1
    max_epochs = int(1e6)

    default_root_dir = f'{tempfile.gettempdir()}/MultiOmics'
    stochastic_weight_avg = False
    limit_train_batches = 100


class MultiOmicsRegressorConfig(BaseModel):
    general_config = GeneralConfig()
    data_config = DataConfig()
    modalities = general_config.modalities

    modalities_model_def = {modality: dict(
        input_features=get_num_attributes(general_config, modality=modality),
        encoder_layer_defs=[
            LayerDef(hidden_dim=8, activation='Hardswish', batch_norm=True),
        ],
        decoder_layer_defs=[
            LayerDef(hidden_dim=None, activation=None, batch_norm=False, layer_type='Dropout', params=dict(p=0.2)),
            LayerDef(hidden_dim=1024, activation='Mish', batch_norm=True),
            LayerDef(hidden_dim=None, activation=None, batch_norm=False, layer_type='Dropout', params=dict(p=0.2)),
            LayerDef(hidden_dim=(get_num_attributes(general_config=general_config, modality=modality)),
                     activation='LeakyReLU', batch_norm=True)
        ],
        regressor_layer_defs=[
            LayerDef(hidden_dim=8, activation='Hardswish', batch_norm=True),
            LayerDef(hidden_dim=None, activation=None, batch_norm=False, layer_type='Dropout', params=dict(p=0.2)),
            LayerDef(hidden_dim=1, activation='ReLU', batch_norm=True)
        ]
    )
        for modality, general_config in zip(modalities, [general_config] * len(modalities))}

    lr = 1e-3
    loss_config = dict(
        triplet_loss=dict(margin=1.0, p=2.0, eps=1e-06, swap=True),
        autoencoding_loss=dict()
    )
    loss_weight_dict = dict(triplet_loss=0.9, autoencoding_loss=0.2)

    short_long_survival_cutoff = data_config.short_long_survival_cutoff

    def __hash__(self):
        return hash(repr(self))


class DataModule(LightningDataModule):
    def __init__(self,
                 train_patients: List[str],
                 val_patients: List[str],
                 collections: List[str],
                 batch_size: int,
                 config_name: str,
                 num_workers=None,
                 *args, **kwargs):
        super().__init__()

        assert not set(train_patients).intersection(set(val_patients))

        self._train_patients = train_patients
        self._val_patients = val_patients
        self._collections = collections

        self._batch_size = batch_size
        self._num_workers = num_workers or min(os.cpu_count(), 8)
        self._config_name = config_name

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        ds = MultiOmicsDataset(patients=self._train_patients,
                               collections=self._collections,
                               get_from_redis=True
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
                               get_from_redis=True

                               )
        return DataLoader(dataset=ds,
                          num_workers=self._num_workers,
                          batch_size=None,
                          sampler=BatchSampler(sampler=RandomSampler(data_source=ds), batch_size=self._batch_size,
                                               drop_last=True),
                          collate_fn=None
                          )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_dataloader()


class MultiOmicsRegressor(LightningModule):
    def __init__(self,
                 modalities: List[str],
                 modalities_model_config: dict,
                 train_patients: List[str],
                 val_patients: List[str],
                 lr: float,
                 loss_config: dict,
                 loss_weight_dict: dict,
                 short_long_survival_cutoff: float,
                 *args: Any,
                 **kwargs: Any):
        super().__init__()

        self.short_long_survival_cutoff = short_long_survival_cutoff
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

        self.input_dropout = nn.Dropout(p=0.05)

        self.smape = SymmetricMeanAbsolutePercentageError()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.step(batch, purpose='train')

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='val')

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='test')

    def configure_optimizers(self):
        return torch.optim.Adam(self.models.parameters(), lr=self.lr)

    def step(self, batch, purpose: str):
        anchor_out = self.models[batch['anchor_modality']](self.input_dropout(batch['anchor']))
        pos_out = self.models[batch['pos_modality']](self.input_dropout(batch['pos']))
        neg_out = self.models[batch['neg_modality']](self.input_dropout(batch['neg']))

        triplet_loss = self.losses['triplet_loss']['fn'](
            anchor_out['encoder'],
            pos_out['encoder'],
            neg_out['encoder']
        )

        self.log(name=f'{purpose}/triplet_loss', value=triplet_loss)

        anchor_rec = self.losses['autoencoding_loss']['fn'](anchor_out['autoencoder'], batch['anchor'])
        self.log(f'{purpose}/{batch["anchor_modality"]}_reg', anchor_rec)
        pos_rec = self.losses['autoencoding_loss']['fn'](pos_out['autoencoder'], batch['pos'])
        self.log(f'{purpose}/{batch["pos_modality"]}_reg', pos_rec)
        neg_rec = self.losses['autoencoding_loss']['fn'](neg_out['autoencoder'], batch['neg'])
        self.log(f'{purpose}/{batch["neg_modality"]}_reg', neg_rec)

        reconstruction_loss = sum((anchor_rec, pos_rec, neg_rec)) / 3
        self.log(f'{purpose}/reconstruction_loss', value=reconstruction_loss)

        cosine_embedding_loss = nn.CosineEmbeddingLoss(margin=0.5)

        pos_embedding_loss = cosine_embedding_loss(
            anchor_out['encoder'], pos_out['encoder'],
            torch.tensor([1] * anchor_out['encoder'].shape[0]).type_as(anchor_out['encoder'])
        )
        neg_embedding_loss = cosine_embedding_loss(
            anchor_out['encoder'], neg_out['encoder'],
            torch.tensor([-1] * anchor_out['encoder'].shape[0]).type_as(anchor_out['encoder'])
        )

        self.log(f'{purpose}/pos_embedding_loss', pos_embedding_loss)
        self.log(f'{purpose}/neg_embedding_loss', neg_embedding_loss)

        regression_loss = mse_loss(
            input=torch.cat(
                (
                    anchor_out['regression'],
                    pos_out['regression'],
                    neg_out['regression'])
            ),
            target=(
                torch.cat(
                    (
                        batch['anchor_survival'],
                        batch['pos_survival'],
                        batch['neg_survival'])
                )
            ).unsqueeze(1)
        )

        self.log(f'{purpose}/survival_regression_loss', regression_loss)
        smape = self.smape(
            preds=torch.cat(
                (
                    anchor_out['regression'],
                    pos_out['regression'],
                    neg_out['regression'])
            ),
            target=(
                torch.cat(
                    (
                        batch['anchor_survival'],
                        batch['pos_survival'],
                        batch['neg_survival'])
                )
            ).unsqueeze(1)
        )

        self.log(f'{purpose}/smape_survival', smape)
        return 5 * pos_embedding_loss + neg_embedding_loss + reconstruction_loss + 3 * regression_loss

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


def load_config_from_file(cls, config_path: str, **kwargs):
    if config_path is not None:
        return cls(**toml.load(Path(config_path).open()), **kwargs)
    return cls(**kwargs)


@app.command()
def train(general_config_path: str = typer.Option(None,

                                                  help=f'Path to a general config file containing any of the '
                                                       f'following keys: {", ".join(GeneralConfig().dict().keys())}'),
          data_config_path: str = typer.Option(None,
                                               help=f'Path to a data config file containing any of the '
                                                    f'following keys: {", ".join(DataConfig().dict().keys())}'),
          trainer_config_path: str = typer.Option(None,
                                                  help=f'Path to a trainer config file containing any of the '
                                                       f'following keys: {", ".join(TrainerConfig().dict().keys())}'),
          multi_omics_regressor_config_path: str = typer.Option(None,
                                                                help=f'Path to a multi_omics_regressor config file '
                                                                     f'containing any of the '
                                                                     f'following keys: {", ".join(MultiOmicsRegressorConfig().dict().keys())}')
          ):
    general_config: GeneralConfig = load_config_from_file(GeneralConfig, config_path=general_config_path)

    data_config: DataConfig = load_config_from_file(DataConfig, config_path=data_config_path,
                                                    general_config=general_config)
    trainer_config: TrainerConfig = load_config_from_file(TrainerConfig, config_path=trainer_config_path)
    multi_omics_regressor_config: MultiOmicsRegressorConfig = load_config_from_file(MultiOmicsRegressorConfig,
                                                                                    config_path=multi_omics_regressor_config_path,
                                                                                    general_config=general_config)

    os.makedirs(Path(trainer_config.default_root_dir, 'wandb').as_posix(), exist_ok=True)

    if not general_config.DEBUG:
        wandb_logger = WandbLogger(name=f"MultiOmics",
                                   log_model='all',
                                   save_dir=trainer_config.default_root_dir
                                   )

        wandb_logger.experiment.config.update(dict(general_config=general_config.dict(),
                                                   data_config=data_config.dict(),
                                                   trainer_config=trainer_config.dict()))
        model_checkpoint = ModelCheckpoint(monitor='val/smape_survival')
    else:
        wandb_logger = None
        model_checkpoint = ModelCheckpoint()

    model = MultiOmicsRegressor(modalities_model_config=multi_omics_regressor_config.modalities_model_def,
                                train_patients=data_config.train_patients,
                                val_patients=data_config.val_patients,
                                **multi_omics_regressor_config.dict(),
                                trainer_config=trainer_config
                                )

    trainer = Trainer(**trainer_config.dict(),
                      logger=[wandb_logger if not general_config.DEBUG else False],
                      callbacks=[LearningRateMonitor(), model_checkpoint],

                      )
    datamodule = DataModule(**data_config.dict())

    trainer.fit(model,
                datamodule=datamodule,
                )


@app.command()
def validate(run_id: str = '2k9d2eog'):
    run = wandb.init(id=run_id)
    weights_file = wandb.restore('epoch=331-step=33199.ckpt')
    model = MultiOmicsRegressor.load_from_checkpoint(weights_file.name)
    data_config: DataConfig = DataConfig(train_patients=tuple(model.hparams.train_patients),
                                         val_patients=tuple(model.hparams.val_patients)
                                         )
    datamodule = DataModule(**data_config.dict())
    trainer_config: TrainerConfig = TrainerConfig()

    trainer = Trainer(**trainer_config.dict())

    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    app()
