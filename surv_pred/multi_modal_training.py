import os
from pathlib import Path
import typing as tp
import pandas as pd
from pymongo import MongoClient
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from surv_pred.datasets import MultiModalityDataset


class MultiModalitiesDataModule(LightningDataModule):
    def __init__(self, batch_size: int, dataset_params: dict, modalities: tp.List[str], **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_params = dataset_params
        with MongoClient(dataset_params['db_params']['mongodb_connection_string']) as client:
            db = client[dataset_params['db_params']['db_name']]
            col = db['metadata']

            _patients = col.find({'split': 'train'}).distinct('patient')

        self.modalities = modalities

        _patients = []
        _patients = list(
            set([_patients.extend(self.get_patients_in_modality(modality=modality)) for modality in self.modalities]))

        _patients = list(set(_patients).intersection(set(_patients)))
        self.train_patients, self.val_patients = train_test_split(_patients, test_size=0.1)

        self.test_patients = list(
            set(col.find({'split': 'test'}).distinct('patient')).intersection(_patients))

    @staticmethod
    def get_patients_in_modality(modality: str, **kwargs):
        df = pd.read_csv(Path(__file__).parent.joinpath(f'{modality}.csv')).set_index('patient').dropna()
        return df.index.tolist()

    def setup(self, stage: tp.Optional[str] = None) -> None:
        self.train_dataset = MultiModalityDataset(patients=self.train_patients, modalities=self.modalities,
                                                  **self.dataset_params)
        self.val_dataset = MultiModalityDataset(patients=self.val_patients, modalities=self.modalities,
                                                **self.dataset_params,
                                                labtrans=self.train_dataset.labtrans,
                                                scaler=self.train_dataset.scaler
                                                )
        self.test_dataset = MultiModalityDataset(patients=self.test_patients, modalities=self.modalities,
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
