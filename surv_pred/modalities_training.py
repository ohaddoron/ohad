import os
import random
import tempfile
import typing as tp

import numpy as np
import pandas as pd

import torchtuples as tt
import wandb
from surv_pred import models

from torch.nn.functional import nll_loss, mse_loss, binary_cross_entropy

from surv_pred.losses import contrastive_loss
from surv_pred.models import MLPVanilla, SurvMLP, SurvAE
from surv_pred.utils import concordance_index
import lifelines
import torch
import typer

from pymongo import MongoClient
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F, BCELoss
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from typer import Typer
from surv_pred.datasets import ModalitiesDataset
import hydra
from omegaconf import DictConfig, OmegaConf


class ModalitiesDataModule(LightningDataModule):
    def __init__(self, batch_size: int, dataset_params: dict, modality: str, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_params = dataset_params
        with MongoClient(dataset_params['db_params']['mongodb_connection_string']) as client:
            db = client[dataset_params['db_params']['db_name']]
            col = db['metadata']
            _patients = [
                'TCGA-04-1331', 'TCGA-04-1332', 'TCGA-04-1336',
                'TCGA-04-1337', 'TCGA-04-1341', 'TCGA-04-1342',
                'TCGA-04-1343', 'TCGA-04-1346', 'TCGA-04-1347',
                'TCGA-04-1348', 'TCGA-04-1349', 'TCGA-04-1350',
                'TCGA-04-1356', 'TCGA-04-1357', 'TCGA-04-1361',
                'TCGA-04-1362', 'TCGA-04-1364', 'TCGA-04-1365',
                'TCGA-04-1367', 'TCGA-04-1514', 'TCGA-04-1517',
                'TCGA-04-1519', 'TCGA-04-1525', 'TCGA-04-1530',
                'TCGA-04-1536', 'TCGA-04-1542', 'TCGA-04-1638',
                'TCGA-04-1646', 'TCGA-04-1648', 'TCGA-04-1649',
                'TCGA-04-1651', 'TCGA-04-1652', 'TCGA-04-1654',
                'TCGA-04-1655', 'TCGA-05-4244', 'TCGA-05-4249',
                'TCGA-05-4250', 'TCGA-05-4382', 'TCGA-05-4384',
                'TCGA-05-4389', 'TCGA-05-4390', 'TCGA-05-4395',
                'TCGA-05-4396', 'TCGA-05-4397', 'TCGA-05-4398',
                'TCGA-05-4402', 'TCGA-05-4403', 'TCGA-05-4405',
                'TCGA-05-4410', 'TCGA-05-4415', 'TCGA-05-4417',
                'TCGA-05-4418', 'TCGA-05-4420', 'TCGA-05-4422',
                'TCGA-05-4424', 'TCGA-05-4425', 'TCGA-05-4426',
                'TCGA-05-4427', 'TCGA-05-4430', 'TCGA-05-4432',
                'TCGA-05-4433', 'TCGA-05-4434', 'TCGA-05-5420',
                'TCGA-05-5423', 'TCGA-05-5425', 'TCGA-05-5428',
                'TCGA-05-5429', 'TCGA-05-5715', 'TCGA-09-0364',
                'TCGA-09-0366', 'TCGA-09-0367', 'TCGA-09-1661',
                'TCGA-09-1662', 'TCGA-09-1665', 'TCGA-09-1666',
                'TCGA-09-1667', 'TCGA-09-1668', 'TCGA-09-1669',
                'TCGA-09-1670', 'TCGA-09-1673', 'TCGA-09-1674',
                'TCGA-09-2044', 'TCGA-09-2045', 'TCGA-09-2048',
                'TCGA-09-2050', 'TCGA-09-2051', 'TCGA-09-2053',
                'TCGA-09-2054', 'TCGA-09-2056', 'TCGA-10-0926',
                'TCGA-10-0927', 'TCGA-10-0928', 'TCGA-10-0930',
                'TCGA-10-0931', 'TCGA-10-0933', 'TCGA-10-0934',
                'TCGA-10-0936', 'TCGA-10-0937', 'TCGA-10-0938']
            _patients = col.find({'split': 'train'}).distinct('patient')

        patients_in_modality = set(
            self.get_patients_in_modality(**dataset_params['db_params'], modality=modality))
        _patients = list(set(_patients).intersection(set(patients_in_modality)))
        self.train_patients, self.val_patients = train_test_split(_patients, test_size=0.1)

        self.test_patients = list(
            set(col.find({'split': 'test'}).distinct('patient')).intersection(patients_in_modality))
        self.modality = modality
        # self.test_patients = self.val_patients

    def get_patients_in_modality(self, mongodb_connection_string: str, db_name: str, modality: str):
        with MongoClient(mongodb_connection_string) as client:
            col = client[db_name][modality]
            return col.distinct('patient')

    def setup(self, stage: tp.Optional[str] = None) -> None:
        self.train_dataset = ModalitiesDataset(patients=self.train_patients, modality=self.modality,
                                               **self.dataset_params)
        self.val_dataset = ModalitiesDataset(patients=self.val_patients, modality=self.modality, **self.dataset_params,
                                             labtrans=self.train_dataset.labtrans,
                                             scaler=self.train_dataset.scaler
                                             )
        self.test_dataset = ModalitiesDataset(patients=self.test_patients, modality=self.modality,
                                              **self.dataset_params,
                                              labtrans=self.train_dataset.labtrans,
                                              scaler=self.train_dataset.scaler
                                              )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 3,
            multiprocessing_context='fork'
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=os.cpu_count() // 3,
            multiprocessing_context='fork'
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

    @property
    def patients(self):
        return dict(train=self.train_patients, val=self.val_patients, test=self.test_patients)


class ModalitiesModel(LightningModule):
    def __init__(self, params: dict, device: int):
        super().__init__()
        self.save_hyperparameters(params)

        self.net = getattr(models, self.hparams.net_params.name)(**self.hparams.net_params)

    def prepare_data(self) -> None:
        return

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.step(batch, purpose='train')

    def validation_step(self, batch, batch_idx) -> tp.Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='val')

    def test_step(self, batch, batch_idx) -> tp.Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='test')

    def step(self, batch, purpose):
        features, durations, events, surv_fns, event_indices = batch['features'], batch['duration'], batch['event'], \
                                                               batch['surv_fn'], batch['event_index']

        surv_outputs, interm_out = self(features)
        loss: torch.Tensor = torch.tensor(0.).type_as(features)
        brier_score: torch.Tensor = torch.tensor(0.).type_as(features)
        for surv_out, surv_fn, event, event_ind in zip(surv_outputs, surv_fns, events, event_indices):
            if event == 0:
                surv_fn = surv_fn[:event_ind]
                surv_out = surv_out[:event_ind]
            loss += getattr(nn.functional, self.hparams.loss_fn)(surv_out, surv_fn)
            brier_score += torch.mean((surv_fn - surv_out) ** 2)
        loss /= len(surv_outputs)
        brier_score /= len(surv_outputs)

        concordance_index = lifelines.utils.concordance_index(durations.cpu().numpy(),
                                                              (surv_outputs >= 0.01).sum(dim=1).detach().cpu().numpy(),
                                                              event_observed=events.cpu().numpy()
                                                              )

        recon_loss = self.compute_reconstruction_loss(
            reconstruction_loss_params=self.hparams.reconstruction_loss_params,
            prediction=interm_out[-1],
            target=features
        )

        c_loss = self.compute_contrastive_loss(interm_out=interm_out, features=features,
                                               **self.hparams.contrastive_loss_params)

        log_dict = {
            f'{purpose}/loss': loss,
            f'{purpose}/concordance_index': concordance_index,
            f'{purpose}/brier_score': brier_score
        }
        if recon_loss is not None:
            loss += recon_loss
            log_dict.update({f'{purpose}/recon_loss': recon_loss})

        if c_loss is not None:
            loss += c_loss
            log_dict.update({f'{purpose}/c_loss': c_loss})

        self.log_dict(log_dict, prog_bar=True, batch_size=features.shape[0])

        return {'loss': loss, **log_dict}

    def compute_contrastive_loss(self, features, interm_out, use: bool, dropout_rate: float):
        if use:
            repr_1 = interm_out[-2]
            repr_2 = self(nn.functional.dropout(features, dropout_rate))[1][-2]
            embs_1, embs_2, target = [], [], []
            for i in range(len(repr_1)):
                # pos
                embs_1.append(repr_1[i])
                embs_2.append(repr_2[i])
                target.append(torch.tensor(1, dtype=torch.long, device=repr_1.device))
                # neg
                embs_1.append(repr_1[1])
                embs_2.append(repr_2[random.choice(list(set(range(len(repr_1))) - {i}))])
                target.append(torch.tensor(0, dtype=torch.long, device=repr_1.device))
            embs_1 = torch.stack(embs_1, dim=0)
            embs_2 = torch.stack(embs_2, dim=0)
            target = torch.stack(target)

            return nn.CosineEmbeddingLoss(margin=0.5)(embs_1, embs_2, target)
        return

    @staticmethod
    def compute_reconstruction_loss(reconstruction_loss_params, prediction, target):
        if not reconstruction_loss_params['use']:
            return
        return getattr(nn.functional, reconstruction_loss_params['method'])(prediction, target)

    def configure_optimizers(self):
        return AdamW(params=self.net.parameters(), lr=5e-4)

    def forward(self, x) -> tp.Any:
        return self.net(x)


def compute_baseline_hazards(model: ModalitiesModel, batch):
    sorting_ind = torch.argsort(batch['duration'])
    features, durations, events = batch['features'][sorting_ind], batch['duration'][sorting_ind], batch['event'][
        sorting_ind]
    model.net.compute_baseline_hazards(input=features, target=(durations, events), max_duration=50)
    return model


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig):
    dm = ModalitiesDataModule(dataset_params=config.db, batch_size=config.batch_size, modality=config.modality)

    print(OmegaConf.to_yaml(config))

    config.train_patients = dm.train_patients
    config.val_patients = dm.val_patients
    config.test_patients = dm.test_patients

    model = ModalitiesModel(params=dict(config),
                            device=config.gpu,
                            )

    if config.debug:
        ml_logger = TensorBoardLogger(save_dir=tempfile.gettempdir(),
                                      name=f'modality={config.modality}'
                                      )

    else:
        ml_logger = WandbLogger(project=config.project,
                                name=f'modality={config.modality}/network={config.net_params.name}/use_recon_loss={config.reconstruction_loss_params.use}/dropout={config.net_params.dropout}',
                                log_model=config.log_model
                                )
        ml_logger.watch(model, log_graph=True)

    trainer = Trainer(gpus=[config.gpu],
                      logger=ml_logger,
                      callbacks=[EarlyStopping(**config.early_stop_monitor)],
                      limit_train_batches=0.2,
                      log_every_n_steps=20
                      )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
