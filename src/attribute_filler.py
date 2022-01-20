import os
import random
import sys
import typing as tp
import warnings
from functools import lru_cache
from pathlib import Path
from typing import *
from src.logger import logger
import torch.cuda
from pydantic import BaseModel
from pytorch_lightning import LightningModule, Trainer, LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torch import nn
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from common.database import init_database
from src.callbacks import WANDBCheckpointCallback
from src.dataset import AttributeFillerDataset
from src.models import LayerDef
from src.models.mlp import MLP

warnings.filterwarnings("ignore")


# mlf_logger = MLFlowLogger(experiment_name="Attribute Filler",
#                           tracking_uri="http://medical001-5.tau.ac.il/mlflow-server/")


@lru_cache
class GeneralConfig(BaseModel):
    """
    General configuration
    """
    COL = 'GeneExpression'
    DEBUG = getattr(sys, 'gettrace', None)() is not None
    DATABASE_CONFIG_NAME = 'omicsdb'
    OVERRIDE_ATTRIBUTES_FILE = True


@lru_cache
class DataConfig(BaseModel):
    """
    Data configuration
    """
    general_config = GeneralConfig()
    config_name = general_config.DATABASE_CONFIG_NAME
    attribute_drop_rate = 0.05
    batch_size = 4
    collection = general_config.COL

    num_workers: int = 0 if general_config.DEBUG else os.cpu_count()

    patients = init_database(general_config.DATABASE_CONFIG_NAME)[general_config.COL].distinct('patient')
    train_patients = random.choices(patients, k=int(len(patients) * 0.9))
    val_patients = list(set(patients) - set(train_patients))


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


class ModelConfig(BaseModel):
    general_config = GeneralConfig()

    attributes = init_database(general_config.DATABASE_CONFIG_NAME)[general_config.COL].distinct('name')
    input_features = len(attributes)

    layers_def = [
        LayerDef(hidden_dim=512, activation='Hardswish', batch_norm=True),
        LayerDef(hidden_dim=512, activation='Mish', batch_norm=True),
        LayerDef(hidden_dim=input_features, activation='LeakyReLU', batch_norm=True)
    ]

    lr = 1e-4


class DataModule(LightningDataModule):
    def __init__(self, train_patients, val_patients, attribute_drop_rate, collection, batch_size, config_name: str,
                 num_workers=None, *args, **kwargs):
        super().__init__()
        self._train_patients = train_patients
        self._val_patients = val_patients
        self._attribute_drop_rate = attribute_drop_rate
        self._collection = collection
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._config_name = config_name

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=AttributeFillerDataset(self._train_patients,
                                           attributes_drop_rate=self._attribute_drop_rate,
                                           collection_name=self._collection,
                                           config_name=self._config_name
                                           ),
            batch_size=self._batch_size,
            num_workers=os.cpu_count() if self._num_workers is None else self._num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=AttributeFillerDataset(self._val_patients,
                                           attributes_drop_rate=self._attribute_drop_rate,
                                           collection_name=self._collection,
                                           config_name=self._config_name
                                           ),
            batch_size=self._batch_size,
            num_workers=os.cpu_count() if self._num_workers is None else self._num_workers,
            shuffle=False
        )


class AttributeFiller(MLP, LightningModule):
    def __init__(self, collection: str, input_features: int, layer_defs: tp.List[LayerDef],
                 lr=1e-4, *args: Any,
                 **kwargs: Any):
        super().__init__(input_features=input_features, layer_defs=layer_defs, *args, **kwargs)

        self._collection = collection
        self._lr = lr

        self.save_hyperparameters()

    def _network_def(self, input_features):
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_features, out_features=self._encoder_hidden_layer_dim
        )

        self.encoder_hidden_layer_batch_norm = nn.BatchNorm1d(num_features=self._encoder_hidden_layer_dim)

        self.encoder_output_layer = nn.Linear(
            in_features=self._encoder_hidden_layer_dim, out_features=self._decoder_hidden_layer_dim
        )

        self.encoder_output_layer_norm = nn.BatchNorm1d(num_features=self._encoder_hidden_layer_dim)

        self.decoder_hidden_layer = nn.Linear(
            in_features=self._decoder_hidden_layer_dim, out_features=self._decoder_hidden_layer_dim
        )

        self.decoder_hidden_layer_norm = nn.BatchNorm1d(num_features=self._decoder_hidden_layer_dim)

        self.decoder_output_layer = nn.Linear(
            in_features=self._decoder_hidden_layer_dim, out_features=input_features
        )

        self.decoder_output_layer_norm = nn.BatchNorm1d(num_features=input_features
                                                        )

        self.net = nn.Sequential(
            self.encoder_hidden_layer,
            self.encoder_hidden_layer_batch_norm,
            nn.Hardswish(),
            self.encoder_output_layer,
            self.encoder_output_layer_norm,
            nn.Mish(),
            self.decoder_hidden_layer,
            self.decoder_hidden_layer_norm,
            nn.Mish(),
            self.decoder_output_layer,
            self.decoder_output_layer_norm,
            nn.LeakyReLU()

        )

    def _get_num_features(self):
        db = init_database('brca-reader')
        return len(db[self._collection].distinct('name'))

    def step(self, batch, purpose: str):
        out = self(batch['attributes'])
        targets = torch.cat([batch['targets'][i][batch['dropped_attributes_index'].long()[i]] for i in
                             range(batch['targets'].shape[0])])
        preds = torch.cat([out[i][batch['dropped_attributes_index'].long()[i]] for i in
                           range(batch['targets'].shape[0])])

        mse_loss = MSELoss()(targets, preds)
        l1_loss = L1Loss()(targets, preds)
        mare = torch.mean((torch.abs(targets - preds)) / (torch.abs(targets) + torch.finfo(torch.float32).eps))

        loss = mse_loss

        # loss = MSELoss()(out, batch['targets'])
        self.log(f'{purpose}/loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log(f'{purpose}/l1', l1_loss, on_step=False, on_epoch=True, logger=True)
        self.log(f'{purpose}/l2', mse_loss, on_step=False, on_epoch=True, logger=True)
        self.log(f'{purpose}/mare', mare, on_step=False, on_epoch=True, logger=True)
        return dict(loss=loss)

    #
    # def forward(self, x) -> Any:
    #     return self.net(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.step(batch=batch, purpose='train')

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.step(batch=batch, purpose='val')

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self._lr)


def main():
    general_config = GeneralConfig()
    data_config = DataConfig()
    trainer_config = TrainerConfig()
    model_config = ModelConfig()
    wandb_logger = WandbLogger("Attribute Filler")

    if general_config.OVERRIDE_ATTRIBUTES_FILE:
        logger.info('Dumping raw attributes to file')
        AttributeFillerDataset.dump_raw_attributes_file(collection=data_config.collection,
                                                        config_name=general_config.DATABASE_CONFIG_NAME,
                                                        output_file=Path(Path(__file__).parent,
                                                                         '../resources/gene_expression_attributes.json').as_posix()
                                                        )

    wandb_logger.experiment.config.update(dict(general_config=general_config.dict(),
                                               data_config=data_config.dict(),
                                               trainer_config=trainer_config.dict()))
    db = init_database(general_config.DATABASE_CONFIG_NAME)
    patients = sorted(db[general_config.COL].distinct('patient'))

    train_patients = sorted(random.choices(patients, k=int(len(patients) * 0.9)))
    val_patients = sorted(list(set(patients) - set(train_patients)))
    model = AttributeFiller(train_patients=train_patients, val_patients=val_patients, collection=general_config.COL,
                            lr=model_config.lr, input_features=model_config.input_features,
                            layer_defs=model_config.layers_def)

    trainer = Trainer(**trainer_config.dict(),
                      logger=[wandb_logger,
                              TensorBoardLogger(
                                  Path(Path(__file__).parent, 'lightning_logs',
                                       name='').as_posix())] if not general_config.DEBUG else False,
                      callbacks=[WANDBCheckpointCallback()],

                      )

    trainer.fit(model,
                datamodule=DataModule(**data_config.dict())
                )


if __name__ == '__main__':
    main()
