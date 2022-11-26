import os
import tempfile
from pathlib import Path
import typing as tp

import hydra
import lifelines
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf, DictConfig
from pymongo import MongoClient
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import Parameter
from torch.optim import AdamW
from torch.utils.data import DataLoader

from surv_pred import models
from surv_pred.datasets import MultiModalityDataset


class MultiModalitiesDataModule(LightningDataModule):
    def __init__(self, batch_size: int, dataset_params: dict, modalities: tp.List[str], **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_params = dataset_params
        with MongoClient(dataset_params['db_params']['mongodb_connection_string']) as client:
            db = client[dataset_params['db_params']['db_name']]
            col = db['metadata']

            train_patients = col.find({'split': 'train'}).distinct('patient')

        self.modalities = modalities

        self.train_patients, self.val_patients = train_test_split(train_patients, test_size=0.1)

        self.test_patients = list(set(col.find({'split': 'test'}).distinct('patient')))

    @staticmethod
    def get_patients_in_modality(modality: str, **kwargs):
        df = pd.read_csv(Path(__file__).parent.joinpath(f'{modality}.csv')).set_index('patient').dropna()
        return df.index.tolist()

    def setup(self, stage: tp.Optional[str] = None) -> None:
        self.train_dataset = MultiModalityDataset(patients=self.train_patients, modalities=self.modalities,
                                                  **self.dataset_params)
        self.val_dataset = MultiModalityDataset(patients=self.val_patients, modalities=self.modalities,
                                                **self.dataset_params,
                                                labtrans={modality: self.train_dataset.datasets[modality].labtrans for
                                                          modality in self.modalities},
                                                scaler={modality: self.train_dataset.datasets[modality].scaler for
                                                        modality in self.modalities}
                                                )
        self.test_dataset = MultiModalityDataset(patients=self.test_patients, modalities=self.modalities,
                                                 **self.dataset_params,
                                                 labtrans={modality: self.train_dataset.datasets[modality].labtrans for
                                                           modality in self.modalities},
                                                 scaler={modality: self.train_dataset.datasets[modality].scaler for
                                                         modality in self.modalities}
                                                 )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 3,
            collate_fn=self.train_dataset.collate_fn,
            multiprocessing_context='fork'
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=os.cpu_count() // 3,
            collate_fn=self.val_dataset.collate_fn,
            multiprocessing_context='fork'
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=0
        )

    @property
    def patients(self):
        return dict(train=self.train_patients, val=self.val_patients, test=self.test_patients)


class MultiModalitiesModel(LightningModule):
    def __init__(self, params: dict):
        super().__init__()
        self.save_hyperparameters(params)

        self.nets = torch.nn.ModuleDict({modality: getattr(models, self.hparams.model_configs[modality].name)(
            **self.hparams.model_configs[modality].net_params) for
            modality in self.hparams.modalities})

        self.survival_threshold = Parameter(torch.tensor(0.5), requires_grad=False)

    def prepare_data(self) -> None:
        return

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.step(batch, purpose='train')

    def validation_step(self, batch, batch_idx) -> tp.Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='val')

    def test_step(self, batch, batch_idx) -> tp.Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='test')

    def step(self, batch, purpose):
        surv_outputs, interm_out, brier_score, concordance_index, loss, recon_loss, c_loss = [{}, {}, {}, {}, {}, {},
                                                                                              {}]
        patients_interm_outputs = {}
        for modality in batch.keys():

            features, durations, events, surv_fns, event_indices = batch[modality]['features'], \
                                                                   batch[modality]['duration'], \
                                                                   batch[modality]['event'], \
                                                                   batch[modality]['surv_fn'], \
                                                                   batch[modality]['event_index']

            surv_outputs[modality], interm_out = self.nets[modality](features)
            
            loss[modality] = torch.tensor(0.).type_as(features)
            brier_score[modality] = torch.tensor(0.).type_as(features)
            for surv_out, surv_fn, event, event_ind in zip(surv_outputs, surv_fns, events, event_indices):
                if event == 0:
                    surv_fn = surv_fn[:event_ind]
                    surv_out = surv_out[:event_ind]
                loss[modality] += getattr(nn.functional, self.hparams.loss_fn)(surv_out, surv_fn)
                brier_score[modality] += torch.mean((surv_fn - surv_out) ** 2)
            loss[modality] /= len(surv_outputs)
            brier_score[modality] /= len(surv_outputs)

            concordance_index[modality] = lifelines.utils.concordance_index(durations.cpu().numpy(),
                                                                            torch.tensor(
                                                                                surv_outputs >= self.survival_threshold).sum(
                                                                                dim=1).detach().cpu().numpy(),
                                                                            event_observed=events.cpu().numpy()
                                                                            )

            recon_loss[modality] = self.compute_reconstruction_loss(
                reconstruction_loss_params=self.hparams.reconstruction_loss_params,
                prediction=interm_out[-1],
                target=features
            )

            c_loss[modality] = self.compute_contrastive_loss(interm_out=interm_out, features=features,
                                                             **self.hparams.contrastive_loss_params)
        pass

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
        self.survival_threshold = Parameter(torch.tensor(thresholds[optimal], requires_grad=False))
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
        return AdamW(params=self.nets.parameters(recurse=True), lr=1e-4)

    def forward(self, x) -> tp.Any:
        return self.net(x)


@hydra.main(version_base=None, config_path='config/multi_modality_config', config_name='config')
def main(config: DictConfig):
    dm = MultiModalitiesDataModule(dataset_params=config.db, batch_size=config.batch_size, modalities=config.modalities)

    print(OmegaConf.to_yaml(config))

    config.train_patients = dm.train_patients
    config.val_patients = dm.val_patients
    config.test_patients = dm.test_patients

    model = MultiModalitiesModel(params=dict(config)
                                 )

    if config.debug:
        ml_logger = TensorBoardLogger(save_dir=tempfile.gettempdir(),
                                      name=f'MultiModality={config.modalities}'
                                      )

    # else:
    #     ml_logger = WandbLogger(project=config.project,
    #                             name=f'modality={config.modality}/network={config.net_params.name}/use_recon_loss={config.reconstruction_loss_params.use}/dropout={config.net_params.dropout}',
    #                             log_model=config.log_model
    #                             )
    #     ml_logger.watch(model, log_graph=True)
    else:
        raise

    trainer = Trainer(gpus=[4],
                      logger=ml_logger,
                      callbacks=[EarlyStopping(**config.early_stop_monitor)],
                      limit_train_batches=0.2,
                      log_every_n_steps=20
                      )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
