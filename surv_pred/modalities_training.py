import os
import typing as tp

from pycox.evaluation.concordance import concordance_td

from surv_pred.utils import concordance_index
import lifelines
import torch
import typer
from pycox.evaluation import EvalSurv
from pycox.models import CoxTime, LogisticHazard
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.models.loss import NLLLogistiHazardLoss
from pymongo import MongoClient
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typer import Typer
from surv_pred.datasets import ModalitiesDataset

app = Typer()

INPUT_NODES = dict(
    mRNA=5000,
    miRNA=1876,
    DNAm=5153
)
CONFIG = {
    'dataset_params':
        {

            'modality': 'miRNA',
           
        },

    'mlp_params': {'in_features': 1876, 'num_nodes': (32, 32), 'batch_norm': True,
                   'activation': nn.ReLU, 'dropout': 0.},
    'batch_size': 32
}


class ModalitiesDataModule(LightningDataModule):
    def __init__(self, batch_size: int, dataset_params: dict, **kwargs):
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
            self.get_patients_in_modality(**dataset_params['db_params'], modality=dataset_params['modality']))
        _patients = list(set(_patients).intersection(set(patients_in_modality)))
        self.train_patients, self.val_patients = train_test_split(_patients, test_size=0.1)

        self.test_patients = list(
            set(col.find({'split': 'test'}).distinct('patient')).intersection(patients_in_modality))
        # self.test_patients = self.val_patients

    def get_patients_in_modality(self, mongodb_connection_string: str, db_name: str, modality: str):
        with MongoClient(mongodb_connection_string) as client:
            col = client[db_name][modality]
            return col.distinct('patient')

    def setup(self, stage: tp.Optional[str] = None) -> None:
        self.train_dataset = ModalitiesDataset(patients=self.train_patients, **self.dataset_params)
        self.val_dataset = ModalitiesDataset(patients=self.val_patients, **self.dataset_params,
                                             feature_names=self.train_dataset.feature_names,
                                             labtrans=self.train_dataset.labtrans,
                                             scaler=self.train_dataset.scaler
                                             )
        self.test_dataset = ModalitiesDataset(patients=self.test_patients, **self.dataset_params,
                                              feature_names=self.train_dataset.feature_names,
                                              labtrans=self.train_dataset.labtrans,
                                              scaler=self.train_dataset.scaler
                                              )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() // 3
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=False,
            num_workers=os.cpu_count() // 3
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
        self.save_hyperparameters()
        self.model: CoxTime = CoxTime(net=MLPVanillaCoxTime(**self.hparams.mlp_params),
                                      device=torch.device(f'cuda:{self.hparams.device}'))
        self.net = self.model.net

        params.update({'patients': self.hparams.patients})

    def prepare_data(self) -> None:
        return

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.step(batch, purpose='train')

    def validation_step(self, batch, batch_idx) -> tp.Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='val')

    def test_step(self, batch, batch_idx) -> tp.Optional[STEP_OUTPUT]:
        return self.step(batch, purpose='test')

    def step(self, batch, purpose):
        features, durations, events = batch['features'], batch['duration'].unsqueeze(1), batch['event'].unsqueeze(1)

        out = self(features, durations)

        self.model.compute_baseline_hazards(input=features, target=(durations.squeeze(), events.squeeze()))

        loss = self.model.loss(out, durations)
        concordance_td(durations=durations.detach().cpu().numpy(),
                       events=events.detach().cpu().numpy(),
                       surv=out.detach().cpu().numpy().T)
        concordance_index_ = lifelines.utils.concordance_index(
            event_times=durations.detach().cpu().numpy(),
            predicted_scores=out.detach().cpu().numpy(),
            event_observed=events.detach().cpu().numpy()
        )

        log_dict = {
            f'{purpose}/loss': loss,
            # f'{purpose}/integrated_brier_score': torch.mean((self.model.predict_surv(features) - events) ** 2),
            f'{purpose}/concordance_index': concordance_index_
        }
        self.log_dict(log_dict, prog_bar=True)

        return {'loss': loss, **log_dict}

    def configure_optimizers(self):
        return AdamW(params=self.model.net.parameters(), lr=5e-4)

    def forward(self, x, durations) -> tp.Any:
        return self.model.predict_net((x, durations), eval_=False, grads=True)


@app.command()
def main(modality: str, gpu: int = 0, num_features: int = None):
    CONFIG['dataset_params'].update({'modality': modality})
    CONFIG['mlp_params'].update({'in_features': num_features or INPUT_NODES[modality]})

    wandb_logger = WandbLogger(name=f'{modality}-surv-pred',
                               log_model=True
                               )

    dm = ModalitiesDataModule(**CONFIG)
    CONFIG.update(dict(patients=dm.patients))
    model = ModalitiesModel(params=CONFIG, device=gpu)
    trainer = Trainer(gpus=[gpu],
                      logger=wandb_logger,
                      callbacks=[EarlyStopping(monitor='val/concordance_index', mode='max', patience=20)]
                      )
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    app()
