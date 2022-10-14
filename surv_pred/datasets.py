import csv
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo.collection
import typer
from loguru import logger
from pycox.models import CoxTime
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from torch.utils.data import Dataset, DataLoader
import typing as tp

from dask import dataframe as dd
from tqdm import tqdm

CONFIG = {
    'patients': ['TCGA-09-2054', 'TCGA-10-0933', 'TCGA-05-4389', 'TCGA-04-1331', 'TCGA-04-1332', 'TCGA-04-1336'],
    'modality': 'miRNA',
    'db_params': {
        'mongodb_connection_string': 'mongodb://admin:mimp1lab@132.66.207.18:80/?authSource=admin&authMechanism=SCRAM-SHA-256&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false',
        'db_name': 'TCGAOmics'
    }

}


def connect_to_database(mongodb_connection_string: str, db_name: str) -> Database:
    return MongoClient(mongodb_connection_string)[db_name]


class ModalitiesDataset(Dataset, ABC):
    def __init__(self, modality: str, patients: tp.List[str], db_params: dict, feature_names=None,
                 labtrans: CoxTime.label_transform = None, scaler: StandardScaler = None):
        self._db_params = db_params

        self.patients = list(set(patients))

        logger.info('Fetching raw data')
        self.data = self.fetch_modality_data(**db_params, patients=list(self.patients), modality=modality)
        logger.info('Raw data fetched')

        logger.info('Fetching survival data')
        self.survival_data_raw = self.fetch_survival_data(**db_params, patients=self.patients)
        self._labtrans = labtrans
        self.survival_data = self.transform_survival_data(labtrans=self.labtrans)

        logger.info('Survival data fetched')
        logger.info('Standardizing data')
        self._scaler = self.get_scaler(scaler)
        logger.info('Data standardized')

    @property
    def labtrans(self):
        return self._labtrans

    def transform_survival_data(self, labtrans: CoxTime.label_transform):
        if labtrans is None:
            labtrans = CoxTime.label_transform()
            labtrans.fit_transform(
                durations=np.array(list(map(lambda x: x['duration'], self.survival_data_raw))),
                events=np.array(list(map(lambda x: x['event'], self.survival_data_raw)))
            )
            self._labtrans = labtrans
        durations, events = self.labtrans.transform(
            durations=np.array(list(map(lambda x: x['duration'], self.survival_data_raw))),
            events=np.array(list(map(lambda x: x['event'], self.survival_data_raw)))
        )
        return [dict(patient=patient, duration=duration, event=event) for patient, duration, event in
                zip(map(lambda x: x['patient'], self.survival_data_raw), durations, events)
                ]

    @property
    def scaler(self):
        return self._scaler

    def get_scaler(self, scaler: StandardScaler) -> StandardScaler:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(self.data)
            return scaler
        else:
            return scaler

    @staticmethod
    def fetch_modality_data(mongodb_connection_string: str, db_name: str, modality, patients: tp.List[str]):
        df = pd.read_csv(Path(__file__).parent.joinpath(f'{modality}.csv')).set_index('patient')
        all_data = df.loc[patients]
        return all_data

    def fetch_survival_data(self, mongodb_connection_string: str, db_name: str, patients: tp.List[str]):
        with MongoClient(mongodb_connection_string) as client:
            col = client[db_name]['survival']
            surv_data = list(col.find({'patient': {'$in': patients}},
                                      {'_id': 0}))
        list(map(lambda x: x.update(dict(duration=x['time'])), surv_data))
        return surv_data

    @property
    def feature_names(self):
        return self.data.columns

    def get_feature_names(self):

        if self._feature_names is None:
            feature_names = set(map(lambda x: x['name'], self.data))
            return feature_names
            for patient in tqdm(self.patients):
                feature_names.intersection(
                    set(map(lambda x: x['name'], filter(lambda x: x['patient'] == patient, self.data))))
            self._feature_names = sorted(list(feature_names))

        return self._feature_names

    def filter_features_from_data(self, data: dd.DataFrame):
        return data

    def filter_patients_from_data(self, data: tp.List[dict]):
        return list(filter(lambda x: x['patient'] in set(self.patients), data))

    def __getitem__(self, item):
        patient = self.patients[item]

        return dict(
            features=self.scaler.transform(pd.DataFrame(self.data.loc[patient]).T).squeeze().astype(np.float32),
            **next(filter(lambda x: x['patient'] == patient, self.survival_data))

        )

    def __len__(self):
        return len(self.patients)
