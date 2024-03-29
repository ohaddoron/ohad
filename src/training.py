import os
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Type
import typing as tp
import json

from pycox.evaluation import EvalSurv
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import pytorch_lightning as pl
import toml
import typer
from pydantic import BaseModel
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from bson import json_util
from src.models import LayerDef
from src.models.mlp import MLP, MultiHeadAutoEncoderRegressor, AutoEncoder
import numpy as np
from src.dataset import AttributesDataset, MultiOmicsAttributesDataset
from torch.nn import TripletMarginWithDistanceLoss, TripletMarginLoss
from typer import Typer
import warnings
from torch.nn import functional as F
from pycox.models import LogisticHazard, DeepHitSingle, BCESurv
from pycox.models.loss import NLLLogistiHazardLoss, BCESurvLoss

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


class MultiOmicsGeneralConfig(BaseModel):
    DEBUG = getattr(sys, 'gettrace', None)() is not None
    DATABASE_CONFIG_NAME = 'TCGAOmics'
    mongodb_connection_string = "mongodb://TCGAManager:MongoDb-eb954cffde2cedf17b22b@132.66.207.18:80/TCGAOmics?authSource=admin"
    modalities: tp.List[str]
    num_attributes: tp.Dict[str, float] = None
    num_durations: int = 20

    def __init__(self, modalities: tp.List[str], features: tp.Dict[str, tp.List[str]], *args, **kwargs):
        super().__init__(modalities=modalities, features=features, *args, **kwargs)

        if features is not None:
            self.num_attributes = {modality: len(
                values) for modality, values in features.items()}
        else:
            self.num_attributes = {modality: MultiOmicsAttributesDataset.get_num_attributes(
                mongodb_connection_string=self.mongodb_connection_string,
                db_name=self.DATABASE_CONFIG_NAME,
                modality=modality) for modality in self.modalities}


class MultiOmicsDataConfig(BaseModel):
    debug: bool = False
    config_name: str
    batch_size: int = 16
    num_workers: int = None
    modalities: tp.List[str]
    num_attributes: tp.Dict[str, float]
    mongodb_connection_string: str
    db_name: str
    feature_set: tp.Dict[str, tp.List[str]] = None
    standardization_values: tp.Dict[str, tp.Dict[str, float]] = None
    drop_rate: tp.Dict[str, float] = None
    drop_rate_base: float = 0.2
    num_durations: int

    def __init__(self, debug: bool, *args, **kwargs):
        super().__init__(debug=debug, **kwargs)
        self.num_workers = 0 if debug else min(os.cpu_count(), 8)
        self.drop_rate = {modality: self.drop_rate_base for modality in self.modalities}

    def __hash__(self):
        return hash(repr(self.dict()))


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
    feature_set: tp.List[str] = None

    def __init__(self, debug: bool, **kwargs):
        super().__init__(**kwargs)
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
            LayerDef(hidden_dim=self.input_features,
                     activation='Sigmoid', batch_norm=True),
        ]


class MultiOmicsNetworkConfig(BaseModel):
    input_features: tp.Dict[str, float]
    modality_model: tp.Dict[str, tp.Dict[str, LayerDef]] = dict()
    embedding_size: int = 32
    num_durations: int = None
    regressor_config: dict = None

    def __init__(self, input_features, num_durations: int, *args, **kwargs):
        super().__init__(input_features=input_features, num_durations=num_durations)

        for modality in input_features.keys():
            self.modality_model[modality] = dict(
                encoder_layer_defs=[
                    LayerDef(
                        hidden_dim=64,
                        activation='LeakyReLU',
                        batch_norm=True
                    ),
                    LayerDef(
                        hidden_dim=None,
                        activation=None,
                        batch_norm=False,
                        layer_type='Dropout',
                        params=dict(p=0.2)
                    ),
                    LayerDef(
                        hidden_dim=self.embedding_size,
                        activation='LeakyReLU',
                        batch_norm=True
                    )
                ],
                decoder_layer_defs=[
                    LayerDef(
                        hidden_dim=self.input_features[modality],
                        activation='Sigmoid', batch_norm=True
                    )
                ],
                input_features=input_features[modality]

            )
        self.regressor_config = dict(
            layer_defs=[
                LayerDef(
                    hidden_dim=self.embedding_size,
                    activation='ReLU',
                    batch_norm=True
                ),
                LayerDef(
                    hidden_dim=self.num_durations,
                    activation=None,
                    batch_norm=False
                )
            ],
            input_features=self.embedding_size
        )


class AttributesFillerConfig(BaseModel):
    network_config: NetworkConfig = None
    loss_term: str = 'SmoothL1Loss'
    lr: float = 1e-3

    def __init__(self, input_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network_config = NetworkConfig(
            input_features=input_features
        )


class MultiOmicsModelConfig(BaseModel):
    reconstruction_loss_term: str = 'SmoothL1Loss'
    network_config: MultiOmicsNetworkConfig = None
    num_durations: int = None

    def __init__(self, input_features, num_durations, *args, **kwargs):
        super().__init__(num_durations=num_durations)
        self.network_config = MultiOmicsNetworkConfig(
            input_features=input_features,
            num_durations=num_durations
        )


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

    stochastic_weight_avg = False

    def __init__(self, debug: bool, *args, **kwargs):
        super().__init__(**kwargs)
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
                 modalities: tp.List[str],
                 feature_set: tp.Dict[str, List[str]] = None,
                 drop_rate: tp.Dict[str, float] = 0.2,
                 batch_size=16,
                 num_workers: int = None,
                 standardization_values: dict = None,
                 num_durations: int = None,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.num_durations = num_durations
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

        self.label_transformer = AttributesDataset.get_label_transformer(
            mongodb_connection_string=mongodb_connection_string,
            db_name=db_name,
            patients=self.train_patients,
            num_durations=num_durations)
        if kwargs.get('debug') and standardization_values is None:
            standardization_values = {}

            standardization_values = {
                modality: {feature_name: dict(min_value=0, max_value=1) for feature_name in
                           MultiOmicsAttributesDataset.get_feature_names(
                               mongodb_connection_string=self.mongodb_connection_string,
                               db_name=self.db_name,
                               modality=modality)} for modality in self.modalities}
        if standardization_values is None:
            self.standardization_values = {modality: MultiOmicsAttributesDataset.get_min_max_values(
                mongodb_connection_string=self.mongodb_connection_string,
                db_name=self.db_name,
                modality=modality,
                patients=self.train_patients
            )
                for modality in self.modalities}
        elif isinstance(standardization_values, dict):
            self.standardization_values = standardization_values
            for modality in modalities:
                if self.standardization_values[modality] is None:
                    self.standardization_values[modality] = MultiOmicsAttributesDataset.get_min_max_values(
                        mongodb_connection_string=mongodb_connection_string,
                        db_name=db_name,
                        modality=modality,
                        patients=self.train_patients)
        else:
            raise TypeError('standardization_values must be of type dict')

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

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        ds = MultiOmicsAttributesDataset(mongodb_connection_string=self.mongodb_connection_string,
                                         db_name=self.db_name,
                                         modalities=self.modalities,
                                         patients=self.train_patients,
                                         drop_rate=self.drop_rate,
                                         standardization_values=self.standardization_values,
                                         num_durations=self.num_durations,
                                         label_transformer=self.label_transformer
                                         )

        return DataLoader(
            dataset=ds,
            drop_last=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.num_workers is not None else os.cpu_count(),
            shuffle=True,
            collate_fn=MultiOmicsAttributesDataset.batch_collate_fn(self.modalities)
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        ds = MultiOmicsAttributesDataset(mongodb_connection_string=self.mongodb_connection_string,
                                         db_name=self.db_name,
                                         modalities=self.modalities,
                                         patients=self.val_patients,
                                         drop_rate=self.drop_rate,
                                         standardization_values=self.standardization_values,
                                         num_durations=self.num_durations,
                                         label_transformer=self.label_transformer
                                         )

        return DataLoader(
            dataset=ds,
            drop_last=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.num_workers is not None else os.cpu_count(),
            shuffle=False,
            collate_fn=MultiOmicsAttributesDataset.batch_collate_fn(self.modalities)
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        ds = MultiOmicsAttributesDataset(mongodb_connection_string=self.mongodb_connection_string,
                                         db_name=self.db_name,
                                         modalities=self.modalities,
                                         patients=self.test_patients,
                                         drop_rate=self.drop_rate,
                                         standardization_values=self.standardization_values,
                                         num_durations=self.num_durations,
                                         label_transformer=self.label_transformer
                                         )

        return DataLoader(
            dataset=ds,
            drop_last=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.num_workers is not None else os.cpu_count(),
            shuffle=False,
            collate_fn=MultiOmicsAttributesDataset.batch_collate_fn(self.modalities)
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
        self.log(f'{phase}/loss', loss, prog_bar=True,
                 on_step=phase == 'train', sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, phase='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, phase='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, phase='test')


class MultiOmicsModel(pl.LightningModule):
    triplet_kinds = ['anchor', 'positive', 'negative']

    def __init__(self,
                 network_config: tp.Dict[str, dict],
                 reconstruction_loss_term: str,
                 data_config: dict = None,
                 **kwargs):
        super().__init__()
        self.nets = nn.ModuleDict()
        for modality, config in network_config['modality_model'].items():
            for key in ['encoder_layer_defs', 'decoder_layer_defs']:
                config[key] = [LayerDef(**item) if isinstance(item, dict) else item for item in config[key]]
            self.nets[modality] = AutoEncoder(**config)
        network_config['regressor_config']['layer_defs'] = [LayerDef(**conf) for conf in
                                                            network_config['regressor_config']['layer_defs']]
        self.surv_model = MLP(**network_config['regressor_config'])
        self.surv_loss = NLLLogistiHazardLoss()

        self.logistic_hazard_model = LogisticHazard(net=self.surv_model, loss=self.surv_loss)
        self.reconstruction_loss_term = reconstruction_loss_term
        self.triplet_loss_term = TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=0.2)
        self.save_hyperparameters()
        self.save_hyperparameters(kwargs)
        self.save_hyperparameters(data_config)

    def configure_optimizers(self):
        opts = [torch.optim.Adam(net.parameters(), lr=1e-3)
                for modality, net in self.nets.items()]
        opts += [torch.optim.Adam(self.surv_model.parameters(), lr=5e-4)]
        return opts

    def step(self, batch: dict, phase: str = 'train'):

        embedding_place_holder = torch.empty(
            max([max(batch[modality]['idx']) for modality in self.nets.keys()]) + 1, 3,
            self.hparams.network_config['embedding_size']).to(
            self.device)
        surv_embeddings = torch.empty(0, self.hparams.network_config['embedding_size']).to(self.device)
        surv_embeddings.requires_grad = True
        durations = []
        events = []

        reconstruction_loss = torch.tensor(0.).to(self.device)

        for modality in batch.keys():
            inputs = batch[modality]['inputs']
            reconstruction_targets = batch[modality]['reconstruction_targets']
            net_outputs = self.nets[modality](inputs, return_aux=True)
            rec_loss: torch.Tensor = getattr(torch.nn, self.reconstruction_loss_term)(reduction='none')(
                reconstruction_targets,
                net_outputs['out'])

            for project_id, loss_val in zip(batch[modality]['project_id'], rec_loss):
                self.log(f'{phase}/reconstruction_loss_by_class/{project_id}', loss_val, on_step=False, on_epoch=True)
            rec_loss = torch.mean(rec_loss)

            self.log(f'{phase}/{modality}_reconstruction_loss', rec_loss, on_step=phase == 'train')

            reconstruction_loss += rec_loss

            embeddings = net_outputs['aux']
            for idx, triplet_kind, embedding in zip(
                    batch[modality]['idx'],
                    batch[modality]['triplet_kind'],
                    embeddings
            ):
                embedding_place_holder[idx, self.triplet_kinds.index(triplet_kind)] = embedding
            surv_embeddings = torch.concat((surv_embeddings, embeddings.detach()))
            durations.append(batch[modality]['survival_data']['duration'])
            events.append(batch[modality]['survival_data']['event'])

        durations = torch.concat(durations)
        events = torch.concat(events)

        surv_prediction = self.surv_model(surv_embeddings)

        surv_loss = self.surv_loss(surv_prediction, durations, events)
        ev = EvalSurv(self.logistic_hazard_model.predict_surv_df(surv_embeddings), durations.cpu().numpy(),
                      events.cpu().numpy(), censor_surv='km')
        concordance_td = ev.concordance_td()
        time_grid = np.linspace(durations.cpu().min(), durations.cpu().max(), 100)
        integrated_brier_score = ev.integrated_brier_score(time_grid=time_grid)
        triplet_loss = self.triplet_loss_term(
            embedding_place_holder[:, 0], embedding_place_holder[:, 1], embedding_place_holder[:, 2])
        self.log(f'{phase}/reconstruction_loss', reconstruction_loss, prog_bar=True, on_step=phase == 'train')
        self.log(f'{phase}/triplet_loss', triplet_loss, prog_bar=True, on_step=phase == 'train')
        self.log(f'{phase}/surv_loss', surv_loss, prog_bar=True, on_step=phase == 'train')
        self.log(f'{phase}/concordance_td', concordance_td, prog_bar=True, on_step=phase == 'train')
        self.log(f'{phase}/integrated_brier_score', integrated_brier_score, prog_bar=True, on_step=phase == 'train')
        return triplet_loss + reconstruction_loss + surv_loss

    def forward(self, x: torch.Tensor, modality: str) -> tp.Dict[str, torch.Tensor]:
        return self.nets[modality](x, return_aux=True)

    def training_step(self, batch, batch_idx, *args, **kwargs):
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
    if not general_config.DEBUG:
        lightning_logger = WandbLogger(
            project="AttributeFiller", log_model=True, name=general_config.modality)
    else:
        lightning_logger = None

    data_config = AttributesFillerDataConfig(config_name=general_config.DATABASE_CONFIG_NAME,
                                             batch_size=batch_size,
                                             modality=modality,
                                             num_attributes=general_config.num_attributes,
                                             mongodb_connection_string=general_config.mongodb_connection_string,
                                             db_name=general_config.DATABASE_CONFIG_NAME,
                                             debug=general_config.DEBUG,
                                             feature_set=feature_set)

    trainer_config = TrainerConfig(debug=general_config.DEBUG, gpus=[gpus])
    attributes_filler_config = AttributesFillerConfig(
        input_features=general_config.num_attributes)

    model = AttributesFillerModel(
        **attributes_filler_config.dict(), **data_config.dict())
    dm = AttributesFillerDataModule(**data_config.dict())

    lightning_logger.watch(model)
    trainer = pl.Trainer(**trainer_config.dict(), logger=lightning_logger,
                         callbacks=[
                             ModelCheckpoint(
                                 dirpath=Path(lightning_logger.experiment.settings.sync_dir).joinpath(
                                     'files').as_posix(),
                                 every_n_epochs=10,
                                 filename=f'{modality}-attribute-model'
                             )
                         ]
                         )

    trainer.fit(model, datamodule=dm)


def get_modality_min_max_values(modality):
    return {item['_id']: {'min_value': item['min_value'], 'max_value': item['max_value']} for item in json_util.loads(
        Path(__file__).parent.joinpath(f'../resources/{modality}_min_max_values.json').open().read())}


def init_data_module(data_config: MultiOmicsDataConfig):
    return MultiOmicsDataModule(**data_config.dict())


@app.command()
def train_multi_omics_model(modalities: tp.List[str] = typer.Option(['mRNA', 'DNAm', 'miRNA']),
                            gpus: int = typer.Option(0),
                            batch_size: int = typer.Option(128),
                            ):
    general_config = MultiOmicsGeneralConfig(
        features=None,
        modalities=modalities,
    )

    data_config = MultiOmicsDataConfig(config_name=general_config.DATABASE_CONFIG_NAME,
                                       batch_size=batch_size,
                                       modalities=modalities,
                                       num_attributes=general_config.num_attributes,
                                       mongodb_connection_string=general_config.mongodb_connection_string,
                                       db_name=general_config.DATABASE_CONFIG_NAME,
                                       debug=general_config.DEBUG,
                                       num_durations=general_config.num_durations,

                                       )

    # trainer_config = TrainerConfig(debug=general_config.DEBUG, gpus=[gpus])
    trainer_config = TrainerConfig(debug=general_config.DEBUG, gpus=[gpus])
    multi_omics_model_config = MultiOmicsModelConfig(
        input_features=general_config.num_attributes,
        num_durations=general_config.num_durations
    )

    model = MultiOmicsModel(
        **multi_omics_model_config.dict(),
        data_config=data_config.dict()
    )
    callbacks = []
    if not general_config.DEBUG:
        lightning_logger = WandbLogger(
            project="MultiOmics",
            log_model=True,
            name='-'.join(sorted(general_config.modalities)),
            save_dir=pl.Trainer().default_root_dir
        )
        lightning_logger.watch(model)
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(lightning_logger.experiment.settings.sync_dir).joinpath(
                    'files').as_posix(),
                every_n_epochs=10,
                filename='multi-omics-model'
            )
        )

    else:
        lightning_logger = None

    trainer = pl.Trainer(**trainer_config.dict(), logger=lightning_logger,
                         callbacks=callbacks
                         )
    dm = init_data_module(data_config=data_config)

    trainer.fit(model, dm)


@app.command()
def restore(run_path: str = typer.Option(..., help='wandb run id to restore the model from'),
            debug: bool = typer.Option(False, help='If True, will setup data module in debug mode')):
    import wandb
    ckpt = torch.load(wandb.restore('multi-omics-model-v1.ckpt', run_path=run_path).name)
    model = MultiOmicsModel(**ckpt['hyper_parameters'])
    model.load_state_dict(ckpt['state_dict'])
    model.freeze()

    data_config = MultiOmicsDataConfig(**ckpt['hyper_parameters']['data_config'], debug=debug)

    trainer = pl.Trainer(**TrainerConfig(debug=debug, enable_checkpointing=False).dict())
    dm = init_data_module(data_config=data_config)
    trainer.test(model, dm.train_dataloader())


if __name__ == '__main__':
    app()
