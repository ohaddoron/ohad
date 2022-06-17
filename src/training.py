import os
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List
import typing as tp
import json
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import toml
import typer
from pydantic import BaseModel
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch

from src.models import LayerDef
from src.models.mlp import MLP, MultiHeadAutoEncoderRegressor

from src.dataset import AttributesDataset, MultiOmicsAttributesDataset

from typer import Typer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
app = Typer()


class GeneralConfig(BaseModel):
    DEBUG = getattr(sys, 'gettrace', None)() is not None
    DATABASE_CONFIG_NAME = 'TCGAOmics'
    mongodb_connection_string = "mongodb://TCGAManager:MongoDb-eb954cffde2cedf17b22b@132.66.207.18:80/TCGAOmics?authSource=admin"
    OVERRIDE_ATTRIBUTES_FILE = True
    modality: str
    num_attributes: int = None

    def __init__(self, modality, features=None, *args, **kwargs):
        super().__init__(modality=modality, features=features, *args, **kwargs)

        if features is not None:
            self.num_attributes = len(features)
        else:
            self.num_attributes = AttributesDataset.get_num_attributes(
                mongodb_connection_string=self.mongodb_connection_string,
                modality=modality,
                db_name=self.DATABASE_CONFIG_NAME
            )

    def __hash__(self):
        return hash(repr(self))


class AttributesFillerDataConfig(BaseModel):
    """
    Data configuration
    """
    config_name: str
    batch_size: int = 32
    num_workers: int = None
    modality: str
    num_attributes: int
    mongodb_connection_string: str
    db_name: str
    feature_set: tp.List[str]

    def __init__(self, debug: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = 0 if debug else min(os.cpu_count(), 16)

    def __hash__(self):
        return hash(repr(self))


class NetworkConfig(BaseModel):
    input_features: int
    layer_defs: tp.List[LayerDef] = None

    def __init__(self, input_features, *args, **kwargs):
        super().__init__(input_features=input_features)
        self.input_features = input_features
        self.layer_defs = [
            LayerDef(hidden_dim=64, activation='LeakyReLU', batch_norm=True),
            LayerDef(hidden_dim=None, activation=None, batch_norm=False, layer_type='Dropout',
                     params=dict(p=0.2)),
            LayerDef(hidden_dim=8, activation='LeakyReLU', batch_norm=True),
            LayerDef(hidden_dim=None, activation=None, batch_norm=False, layer_type='Dropout',
                     params=dict(p=0.2)),
            LayerDef(hidden_dim=self.input_features, activation='Sigmoid', batch_norm=True),
        ]


class AttributesFillerConfig(BaseModel):
    network_config: NetworkConfig = None
    loss_term: str = 'SmoothL1Loss'
    lr: float = 1e-3

    def __init__(self, input_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network_config = NetworkConfig(input_features=input_features)


class TrainerConfig(BaseModel):
    """
    Trainer configuration
    """
    reload_dataloaders_every_n_epochs = False

    enable_checkpointing = True

    profiler = 'simple'
    fast_dev_run: bool = None
    progress_bar_refresh_rate = 1
    max_epochs = int(1e6)
    gpus: tp.Union[int, tp.List[int]] = 1

    stochastic_weight_avg = True

    def __init__(self, debug: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_dev_run = debug


class AttributesFillerDataModule(pl.LightningDataModule):
    def __init__(self,
                 mongodb_connection_string: str,
                 db_name: str,
                 modality: str,
                 feature_set: List[str] = None,
                 drop_rate: float = 0.2,
                 batch_size=32,
                 num_workers: int = None,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.mongodb_connection_string = mongodb_connection_string
        self.db_name = db_name
        self.modality = modality
        self.features_set = feature_set
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_patients, self.val_patients = AttributesDataset.get_train_val_split(
            mongodb_connection_string=self.mongodb_connection_string,
            db_name=self.db_name,
            metadata_collection_name='metadata'
        ).values()
        self.standardization_values = AttributesDataset.get_min_max_values(
            mongodb_connection_string=self.mongodb_connection_string,
            db_name=self.db_name,
            modality=self.modality,
            patients=self.train_patients
        )

        self.test_patients = AttributesDataset.get_test_patients(
            mongodb_connection_string=self.mongodb_connection_string,
            db_name=self.db_name,
            metadata_collection_name='metadata'
        )
        self.save_hyperparameters(
            dict(train_patients=self.train_patients,
                 val_patients=self.val_patients,
                 test_patients=self.test_patients)
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        ds = AttributesDataset(mongodb_connection_string=self.mongodb_connection_string,
                               db_name=self.db_name,
                               modality=self.modality,
                               patients=self.train_patients,
                               features=self.features_set,
                               drop_rate=self.drop_rate,
                               standardization_values=self.standardization_values
                               )

        return DataLoader(
            dataset=ds,
            drop_last=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers or os.cpu_count(),
            shuffle=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        ds = AttributesDataset(
            mongodb_connection_string=self.mongodb_connection_string,
            db_name=self.db_name,
            modality=self.modality,
            patients=self.val_patients,
            features=self.features_set,
            drop_rate=self.drop_rate,
            standardization_values=self.standardization_values
        )

        return DataLoader(
            dataset=ds,
            drop_last=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers or os.cpu_count(),
            shuffle=False
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        ds = AttributesDataset(
            mongodb_connection_string=self.mongodb_connection_string,
            db_name=self.db_name,
            modality=self.modality,
            patients=self.test_patients,
            features=self.features_set,
            drop_rate=self.drop_rate,
            standardization_values=self.standardization_values
        )

        return DataLoader(
            dataset=ds,
            drop_last=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers or os.cpu_count(),
            shuffle=False
        )


class MultiOmicsDataModule(pl.LightningDataModule):
    def __init__(self,
                 mongodb_connection_string: str,
                 db_name: str,
                 modalities: str,
                 feature_set: tp.Dict[str, List[str]] = None,
                 drop_rate: tp.Dict[str, float] = 0.2,
                 batch_size=256,
                 num_workers: int = None,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.mongodb_connection_string = mongodb_connection_string
        self.db_name = db_name
        self.modalities = modalities
        self.features_set = feature_set
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_patients, self.val_patients = MultiOmicsAttributesDataset.get_train_val_split(
            mongodb_connection_string=self.mongodb_connection_string,
            db_name=self.db_name,
            metadata_collection_name='metadata'
        ).values()
        self.standardization_values = {modality: MultiOmicsAttributesDataset.get_min_max_values(
            mongodb_connection_string=self.mongodb_connection_string,
            db_name=self.db_name,
            modality=modality,
            patients=self.train_patients
        )
            for modality in self.modalities}

        self.test_patients = MultiOmicsAttributesDataset.get_test_patients(
            mongodb_connection_string=self.mongodb_connection_string,
            db_name=self.db_name,
            metadata_collection_name='metadata'
        )
        self.save_hyperparameters(
            dict(train_patients=self.train_patients,
                 val_patients=self.val_patients,
                 test_patients=self.test_patients)
        )


class AttributesFillerModel(pl.LightningModule):
    def __init__(self, network_config: dict, lr: float, loss_term: str, *args, **kwargs):
        super().__init__()
        self.network_config = network_config
        self.lr = lr
        self.save_hyperparameters()
        self.loss_term = loss_term
        self.net = MLP(input_features=network_config['input_features'],
                       layer_defs=[LayerDef(**item) for item in network_config['layer_defs']])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def step(self, batch, phase: str = 'train'):
        pred = self.net(batch['inputs'])

        loss_term = getattr(torch.nn, self.loss_term)()
        loss = loss_term(pred, batch['outputs'])
        self.log(f'{phase}/loss', loss, prog_bar=True, on_step=phase == 'train', sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, phase='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, phase='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, phase='test')


class MultiOmicsModel(pl.LightningModule):
    triplet_kinds = ['anchor', 'positive', 'negative']

    def __init__(self, network_configs: tp.Dict[str, dict], reconstruction_loss_term: str, *args, **kwargs):
        self.nets = nn.ModuleDict(
            {modality: MultiHeadAutoEncoderRegressor(**config) for modality, config in network_configs.items()})

        self.reconstruction_loss_term = reconstruction_loss_term
        self.save_hyperparameters()

    def configure_optimizers(self):
        opts = [torch.optim.Adam(net.parameters(), lr=1e-3) for modality, net in self.nets.items()]
        return opts

    def step(self, batch: list, phase: str = 'train'):
        modality_inputs = {modality: dict(inputs=[], reconstruction_targets=[], idx=[]) for modality in
                           self.nets.keys()}
        for i, items in enumerate(batch):
            for j, item in items:
                modality_inputs[item['modality']]['inputs'].append(item['inputs'])
                modality_inputs[item['modality']]['reconstruction_targets'].append(item['outputs'])
                modality_inputs[item['modality']]['idx'].append(i)
                modality_inputs[item['modality']]['kind'].append(self.triplet_kinds[j])

    def training_step(self, batch, batch_idx):
        return self.step(batch, phase='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, phase='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, phase='test')


@app.command()
def train_attributes_filler(modality: str = typer.Option(...),
                            gpus: int = typer.Option(0),
                            batch_size: int = typer.Option(32),
                            features_file_path: str = typer.Option(None)
                            ):
    if features_file_path:
        feature_set = json.load(Path(features_file_path).open())
    else:
        feature_set = None

    general_config = GeneralConfig(modality=modality, features=feature_set)

    wandb_logger = WandbLogger(project="AttributeFiller", log_model=True, name=general_config.modality)

    data_config = AttributesFillerDataConfig(config_name=general_config.DATABASE_CONFIG_NAME,
                                             batch_size=batch_size,
                                             modality=modality,
                                             num_attributes=general_config.num_attributes,
                                             mongodb_connection_string=general_config.mongodb_connection_string,
                                             db_name=general_config.DATABASE_CONFIG_NAME,
                                             debug=general_config.DEBUG,
                                             feature_set=feature_set)

    trainer_config = TrainerConfig(debug=general_config.DEBUG, gpus=[gpus])
    attributes_filler_config = AttributesFillerConfig(input_features=general_config.num_attributes)

    model = AttributesFillerModel(**attributes_filler_config.dict(), **data_config.dict())
    dm = AttributesFillerDataModule(**data_config.dict())

    wandb_logger.watch(model)
    trainer = pl.Trainer(**trainer_config.dict(), logger=wandb_logger)

    trainer.fit(model, datamodule=dm)


@app.command
def train_multi_omics_model(modalities: tp.List = typer.Option(['mRNA', 'DNAm', 'miRNA']),
                            gpus: int = typer.Option(0),
                            batch_size: int = typer.Option(256),
                            ):
    general_config = dict()
    data_config = dict()

    # general_config = GeneralConfig(modality=modality, features=feature_set)

    wandb_logger = WandbLogger(project="AttributeFiller", log_model=True, name=general_config.modality)

    data_config = AttributesFillerDataConfig(config_name=general_config.DATABASE_CONFIG_NAME,
                                             batch_size=batch_size,
                                             modality=modality,
                                             num_attributes=general_config.num_attributes,
                                             mongodb_connection_string=general_config.mongodb_connection_string,
                                             db_name=general_config.DATABASE_CONFIG_NAME,
                                             debug=general_config.DEBUG,
                                             feature_set=feature_set)

    trainer_config = TrainerConfig(debug=general_config.DEBUG, gpus=[gpus])
    attributes_filler_config = AttributesFillerConfig(input_features=general_config.num_attributes)

    model = AttributesFillerModel(**attributes_filler_config.dict(), **data_config.dict())
    dm = AttributesFillerDataModule(**data_config.dict())


if __name__ == '__main__':
    app()
