import csv
import random
from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import pymongo.collection
import torch
import typer
from loguru import logger
from pycox.models import CoxTime
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from torch.utils.data import Dataset, DataLoader
import typing as tp

from dask import dataframe as dd
from tqdm import tqdm

CONFIG = {
    'patients': ['TCGA-09-2054', 'TCGA-10-0933', 'TCGA-05-4389', 'TCGA-04-1331', 'TCGA-04-1332', 'TCGA-04-1336'],
    'modality_update': 'miRNA',
    'db_params': {
        'mongodb_connection_string': 'mongodb://admin:mimp1lab@132.66.207.18:80/?authSource=admin&authMechanism=SCRAM-SHA-256&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false',
        'db_name': 'TCGAOmics'
    }

}


def connect_to_database(mongodb_connection_string: str, db_name: str) -> Database:
    return MongoClient(mongodb_connection_string)[db_name]


class DummyScaler(StandardScaler):
    def fit(self):
        return

    def transform(self, X, copy=None):
        return X


class ModalitiesDataset(Dataset, ABC):
    def __init__(self,
                 modality: str,
                 patients: tp.List[str],
                 db_params: dict,
                 labtrans: CoxTime.label_transform = None,
                 scaler: ColumnTransformer = None,
                 max_survival_duration: float = 32,
                 survival_resolution: int = 100,
                 ):
        self._db_params = db_params
        self.modality = modality

        logger.info(f'Fetching raw data {modality}')
        self.data: pd.DataFrame = self.fetch_modality_data(**db_params, patients=patients, modality=modality)
        self.patients = self.data.index.tolist()
        logger.info('Raw data fetched')

        logger.info('Fetching survival data')
        self.survival_data_raw = self.fetch_survival_data(**db_params, patients=self.patients)
        self._labtrans = labtrans
        self.survival_data = self.transform_survival_data(labtrans=self.labtrans)

        logger.info('Survival data fetched')
        logger.info('Standardizing data')
        self._scaler = self.get_scaler(scaler)
        logger.info('Data standardized')

        self.survival_array = np.linspace(0, max_survival_duration, survival_resolution)

    @property
    def labtrans(self):
        return self._labtrans

    def transform_survival_data(self, labtrans: CoxTime.label_transform):
        if labtrans is None:
            labtrans = CoxTime.label_transform(with_mean=False, with_std=False)
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
    def scaler(self) -> ColumnTransformer:
        return self._scaler

    def get_scaler(self, scaler: ColumnTransformer) -> ColumnTransformer:
        if scaler is None:

            scaler = make_column_transformer(
                (
                    StandardScaler(),
                    make_column_selector(dtype_include=float, dtype_exclude=np.int64),

                ),
                remainder='passthrough'
            )

            scaler.fit(self.data)
            return scaler
        else:
            return scaler

    @staticmethod
    def fetch_modality_data(modality,
                            patients: tp.List[str], **kwargs) -> pd.DataFrame:
        df = pd.read_csv(Path(__file__).parent.joinpath(f'{modality}.csv')).set_index('patient')
        patients = set(df.index.tolist()).intersection(patients)
        all_data = df.loc[patients]
        return all_data.dropna()

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

        return self._feature_names

    def filter_features_from_data(self, data: dd.DataFrame):
        return data

    def filter_patients_from_data(self, data: tp.List[dict]):
        return list(filter(lambda x: x['patient'] in set(self.patients), data))

    def __getitem__(self, item):
        patient = self.patients[item]
        return self.get_patient_data(patient=patient)

    def get_patient_data(self, patient):
        surv_data = next(filter(lambda x: x['patient'] == patient, self.survival_data))

        event_index = np.searchsorted(self.survival_array, surv_data['duration']) + 1
        surv_fn = np.ones(len(self.survival_array), dtype=np.float32)
        if surv_data['event'] == 1.:
            surv_fn[event_index:] = 0.
        if 'Clinical' not in self.modality:
            return dict(
                features=self.scaler.transform(pd.DataFrame(self.data.loc[patient]).T).squeeze().astype(np.float32),
                surv_fn=surv_fn,
                event_index=event_index,
                **surv_data)
        df = pd.DataFrame(self.data.loc[patient]).T
        df = df[['project_id', 'synchronous_malignancy', 'ajcc_pathologic_stage',
                 'tissue_or_organ_of_origin', 'primary_diagnosis', 'prior_malignancy', 'prior_treatment',
                 'ajcc_pathologic_t', 'ajcc_pathologic_n', 'ajcc_pathologic_m', 'site_of_resection_or_biopsy',
                 'race', 'gender', 'treatments_pharmaceutical_treatment_type',
                 'treatments_pharmaceutical_treatment_or_therapy',
                 'treatments_radiation_treatment_type', 'treatments_radiation_treatment_or_therapy', 'figo_stage',
                 'ajcc_clinical_t', 'ajcc_clinical_m', 'age_at_diagnosis', 'pack_years_smoked', 'cigarettes_per_day']]
        return dict(
            features=df.squeeze().astype(np.float32).values,
            surv_fn=surv_fn,
            event_index=event_index,
            **surv_data)

    def __len__(self):
        return len(self.patients)


class MultiModalityDataset(Dataset):
    def __init__(self,
                 modalities: tp.List[str],
                 patients: tp.List[str],
                 db_params=None,
                 labtrans: tp.Dict[str, CoxTime.label_transform] = None,
                 scaler: tp.Dict[str, ColumnTransformer] = None,
                 max_survival_duration: float = 32,
                 survival_resolution: int = 100,
                 fetch_all_modalities=False):
        if db_params is None:
            db_params = {}

        if labtrans is None:
            labtrans = {modality: None for modality in modalities}

        if scaler is None:
            scaler = {modality: None for modality in modalities}

        self.datasets: tp.Dict[str, ModalitiesDataset] = {
            modality: ModalitiesDataset(modality=modality, patients=patients, db_params=db_params,
                                        labtrans=labtrans[modality], scaler=scaler[modality],
                                        max_survival_duration=max_survival_duration,
                                        survival_resolution=survival_resolution) for modality in
            modalities}

        self.patients = patients
        self.fetch_all_modalities = fetch_all_modalities

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient = self.patients[item]

        modalities_available = [key for key in self.datasets.keys() if patient in self.datasets[key].patients]
        if self.fetch_all_modalities:
            chosen_modalities = modalities_available
        else:
            chosen_modalities = random.sample(population=modalities_available, k=min(2, len(modalities_available)))

        out = {f'{modality}/{key}': value for modality in chosen_modalities for key, value in
               self.datasets[modality].get_patient_data(patient=patient).items()
               }
        [out.update({f'{modality}/patient': patient}) for modality in chosen_modalities]
        return out

    @property
    def modalities_available(self):
        return list(key for key in self.datasets.keys())

    def collate_fn(self, batch: tp.List[dict]):
        outputs = {key: {} for key in self.modalities_available}

        for item in batch:
            for key, value in item.items():
                modality, value_type = key.split('/')
                if value_type not in outputs[modality]:
                    outputs[modality][value_type] = []
                outputs[modality][value_type].append(value)

        for key in outputs:
            if not outputs[key]:
                continue
            outputs[key]['features'] = torch.tensor(np.vstack(outputs[key]['features']))
            outputs[key]['surv_fn'] = torch.tensor(np.vstack(outputs[key]['surv_fn']))
            outputs[key]['duration'] = torch.tensor(np.vstack(outputs[key]['duration']))
        return outputs


def multi_modal_config():
    return {
        'patients': ['TCGA-04-1331', 'TCGA-04-1332', 'TCGA-04-1336',
                     'TCGA-04-1337', 'TCGA-04-1341', 'TCGA-04-1342',
                     'TCGA-04-1343', 'TCGA-04-1346', 'TCGA-04-1347',
                     'TCGA-04-1348', 'TCGA-04-1349', 'TCGA-04-1350',
                     ],
        'modalities': ['DNAm', 'mRNA', 'CNV', 'Clinical', 'miRNA'],
        'db_params': {
            'mongodb_connection_string': 'mongodb://admin:mimp1lab@132.66.207.18:80/?authSource=admin&authMechanism=SCRAM-SHA-256&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false',
            'db_name': 'TCGAOmics'
        }

    }


def multi_modality_dataset():
    return MultiModalityDataset(**multi_modal_config())
