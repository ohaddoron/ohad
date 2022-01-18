import os
import random
import sys
from pathlib import Path

import mlflow.pytorch
import torch.cuda
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger, TestTubeLogger, TensorBoardLogger, WandbLogger
from typing import *
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torch import nn
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from common.database import init_database
from src.dataset import AttributeFillerDataset

import warnings

warnings.filterwarnings("ignore")
# mlf_logger = MLFlowLogger(experiment_name="Attribute Filler",
#                           tracking_uri="http://medical001-5.tau.ac.il/mlflow-server/")

wandb_logger = WandbLogger("Attribute Filler")
DEBUG = getattr(sys, 'gettrace', None)() is not None

COL = 'GeneExpression'


class AttributeFiller(LightningModule):
    def __init__(self, train_patients: List[str], val_patients: List[str], collection: str, attribute_drop_rate=0.05,
                 batch_size=4, lr=1e-4, encoder_hidden_layer_dim=128,
                 decoder_hidden_layer_dim=128, *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._train_patients = train_patients
        self._encoder_hidden_layer_dim = encoder_hidden_layer_dim
        self._decoder_hidden_layer_dim = decoder_hidden_layer_dim
        self._val_patients = val_patients
        self._attribute_drop_rate = attribute_drop_rate
        self._collection = collection
        self._batch_size = batch_size
        self._lr = lr

        self._network_def(self._get_num_features())

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

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=AttributeFillerDataset(self._train_patients, attributes_drop_rate=self._attribute_drop_rate,
                                           collection_name=self._collection),
            batch_size=self._batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=AttributeFillerDataset(self._val_patients, attributes_drop_rate=self._attribute_drop_rate,
                                           collection_name=self._collection),
            batch_size=self._batch_size,
            num_workers=os.cpu_count(),
            shuffle=False
        )

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

    def forward(self, x) -> Any:
        return self.net(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.step(batch=batch, purpose='train')

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        return self.step(batch=batch, purpose='val')

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self._lr)


class WANDBModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        out = super().on_save_checkpoint(trainer, pl_module, checkpoint)
        if self.last_model_path:
            wandb.save(self.last_model_path)
        return out


def main():
    db = init_database('brca-reader')
    patients = db[COL].distinct('patient')

    train_patients = random.choices(patients, k=int(len(patients) * 0.9))
    val_patients = list(set(patients) - set(train_patients))
    model = AttributeFiller(train_patients=train_patients, val_patients=val_patients, collection=COL)

    checkpoint_path = 'lightning_logs/version_10/checkpoints/epoch=3194-step=146969.ckpt'

    trainer = Trainer(
        # logger=mlf_logger if not DEBUG else False,
        gpus=1 if torch.cuda.is_available() else None,
        auto_select_gpus=True, accumulate_grad_batches=4,
        reload_dataloaders_every_epoch=False, max_epochs=int(1e5),
        callbacks=[WANDBModelCheckpoint()],
        logger=[wandb_logger,
                TensorBoardLogger(
                    Path(Path(__file__).parent, 'lightning_logs', name='').as_posix())] if not DEBUG else False,
        checkpoint_callback=True,
        resume_from_checkpoint=checkpoint_path if Path(checkpoint_path).exists() and not DEBUG else None
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
