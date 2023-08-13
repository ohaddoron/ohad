import os
import random
import tempfile
import typing as tp
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor
import hydra
import lifelines
import lifelines.utils
import numpy as np
import pandas as pd
import torch
from loguru import logger as LOGGER
from omegaconf import DictConfig, OmegaConf
from pymongo import MongoClient
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import Parameter
from torch.optim import AdamW
from torch.utils.data import DataLoader

from surv_pred import models
from surv_pred.datasets import ModalitiesDataset
from surv_pred.models import SurvMLP


class ModalitiesDataModule(LightningDataModule):
    def __init__(self, batch_size: int, dataset_params: dict, modality: str, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_params = dataset_params
        with MongoClient(dataset_params['db_params']['mongodb_connection_string']) as client:
            db = client[dataset_params['db_params']['db_name']]
            col = db['metadata']

            _patients = col.find({'split': 'train'}).distinct('patient')

        patients_in_modality = set(
            self.get_patients_in_modality(modality=modality))
        _patients = list(set(_patients).intersection(
            set(patients_in_modality)))
        self.train_patients, self.val_patients = train_test_split(
            _patients, test_size=0.1)

        self.test_patients = list(
            set(col.find({'split': 'test'}).distinct('patient')).intersection(patients_in_modality))
        self.modality = modality
        # self.test_patients = self.val_patients

    @staticmethod
    def get_patients_in_modality(modality: str, **kwargs):
        df = pd.read_csv(Path(__file__).parent.joinpath(
            f'{modality}.csv')).set_index('patient').dropna()
        return df.index.tolist()

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
            num_workers=2,
            multiprocessing_context='fork',
            drop_last=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=0,

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
        net_params = dict(self.hparams.net_params)
        name = net_params.pop('name')
        net_params.pop('in_features')
        try:
            net_params['cat_idxs'] = list(net_params['cat_idxs'])
            net_params['cat_dims'] = list(net_params['cat_dims'])
            net_params['cat_emb_dim'] = list(net_params['cat_emb_dim'])
        except KeyError:
            pass
        self.net = getattr(models, name)(
            **net_params).cpu()

        self.survival_threshold = Parameter(
            torch.tensor(0.01), requires_grad=False)

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
            loss += getattr(nn.functional,
                            self.hparams.loss_fn)(surv_out, surv_fn)
            brier_score += torch.mean((surv_fn - surv_out) ** 2)
        loss /= len(surv_outputs)
        brier_score /= len(surv_outputs)
        try:
            concordance_index = lifelines.utils.concordance_index(durations.cpu().numpy(),
                                                                  torch.tensor(
                                                                      surv_outputs >= self.survival_threshold).sum(
                                                                      dim=1).detach().cpu().numpy(),
                                                                  event_observed=events.cpu().numpy()
                                                                  )
        except:
            concordance_index = None
        if interm_out is not None:
            recon_loss = self.compute_reconstruction_loss(
                reconstruction_loss_params=self.hparams.reconstruction_loss_params,
                prediction=interm_out[-1],
                target=features
            )

            c_loss = self.compute_contrastive_loss(interm_out=interm_out, features=features,
                                                   **self.hparams.contrastive_loss_params)
        else:
            recon_loss = torch.tensor(0.).to(self.device)
            c_loss = torch.tensor(0.).to(self.device)

        log_dict = {
            f'{purpose}/loss': loss,

            f'{purpose}/brier_score': brier_score
        }
        if concordance_index is not None:
            log_dict.update({f'{purpose}/concordance_index': concordance_index})
        if recon_loss is not None:
            loss += recon_loss
            log_dict.update({f'{purpose}/recon_loss': recon_loss})

        if c_loss is not None:
            loss += c_loss
            log_dict.update({f'{purpose}/c_loss': c_loss})

        self.log_dict(log_dict, prog_bar=True, batch_size=features.shape[0])

        return {'loss': loss, **log_dict, 'durations': durations.detach(), 'surv_outputs': surv_outputs.detach(),
                'events_observed': events.detach()}

    def training_epoch_end(self, outputs: dict) -> None:
        durations = torch.cat([item['durations'] for item in outputs])
        surv_outputs = torch.cat([item['surv_outputs'] for item in outputs])
        events = torch.cat([item['events_observed'] for item in outputs])
        concordance_index_scores = []
        thresholds = list(torch.arange(0., 1., 0.05))
        for thresh in thresholds:
            concordance_index_scores.append(lifelines.utils.concordance_index(durations.cpu().numpy(),
                                                                              torch.tensor(
                                                                                  surv_outputs >= thresh).sum(
                                                                                  dim=1).detach().cpu().numpy(),
                                                                              event_observed=events.cpu().numpy()
                                                                              )
                                            )
        optimal = np.argmax(concordance_index_scores)
        if self.hparams.calibrate_survival_threshold:
            self.survival_threshold = Parameter(
                torch.tensor(thresholds[optimal], requires_grad=False))
        self.log('survival_threshold', self.survival_threshold)
        # LOGGER.info('Optimal threshold: {}'.format(self.survival_threshold))

    def compute_contrastive_loss(self, features, interm_out, use: bool, dropout_rate: float):
        if use:
            repr_1 = interm_out[-2]
            repr_2 = self(nn.functional.dropout(features, dropout_rate))[1][-2]
            embs_1, embs_2, target = [], [], []
            for i in range(len(repr_1)):
                # pos
                embs_1.append(repr_1[i])
                embs_2.append(repr_2[i])
                target.append(torch.tensor(
                    1, dtype=torch.long, device=repr_1.device))
                # neg
                embs_1.append(repr_1[1])
                embs_2.append(repr_2[random.choice(
                    list(set(range(len(repr_1))) - {i}))])
                target.append(torch.tensor(
                    0, dtype=torch.long, device=repr_1.device))
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
        optimizer = AdamW(params=self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if not self.hparams.use_scheduler:
            return optimizer
        else:
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler.name)(
                optimizer=optimizer, **self.hparams.scheduler.params)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x) -> tp.Any:
        return self.net(x)


def compute_baseline_hazards(model: ModalitiesModel, batch):
    sorting_ind = torch.argsort(batch['duration'])
    features, durations, events = batch['features'][sorting_ind], batch['duration'][sorting_ind], batch['event'][
        sorting_ind]
    model.net.compute_baseline_hazards(
        input=features, target=(durations, events), max_duration=50)
    return model


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig):
    dm = ModalitiesDataModule(
        dataset_params=config.db, batch_size=config.batch_size, modality=config.modality)

    print(OmegaConf.to_yaml(config))

    config.train_patients = dm.train_patients
    config.val_patients = dm.val_patients
    config.test_patients = dm.test_patients

    model = ModalitiesModel(params=dict(config),
                            device=config.gpu,
                            )

    if config.debug:
        ml_logger = TensorBoardLogger(save_dir=tempfile.gettempdir(),
                                      name=f'modal  ity={config.modality}'
                                      )

    else:
        ml_logger = WandbLogger(project=config.project,
                                name=f'modality={config.modality}/network={config.net_params.name}/use_recon_loss={config.reconstruction_loss_params.use}/dropout=0',
                                log_model=config.log_model
                                )
        ml_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor='val/concordance_index', mode="max")

    trainer = Trainer(gpus=[config.gpu],
                      logger=ml_logger,
                      callbacks=[EarlyStopping(
                          **config.early_stop_monitor), LearningRateMonitor('epoch'), checkpoint_callback],
                      # limit_train_batches=0.2,
                      log_every_n_steps=20,
                      gradient_clip_val=config.gradient_clip_val
                      )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
