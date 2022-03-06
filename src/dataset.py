import json
import math
import random
import tempfile
import time
import typing as tp
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import diskcache
import numpy as np
import torch
from numpy.core._simd import targets
from torch.utils.data import Dataset

from common.database import init_database
from src.logger import logger
from src.cache import cache

cache: diskcache.Cache


def time_elapsed(fn, *args, **kwargs):
    start = time.time()
    out = fn(*args, **kwargs)
    elapsed = time.time() - start
    print(f'\n{elapsed}')
    return elapsed


class BaseDataset:
    def __init__(self, patients: tp.List[str], config_name: str = 'omicsdb', get_from_redis: bool = False):
        self.config_name = config_name
        self.patients = self.process_patients(patients)
        self.samples = self.define_samples()
        self._get_from_redis = get_from_redis
        if get_from_redis:
            import redis
            self._redis = redis.Redis(
                host='localhost',
                port=6379,
            )
            if not self._redis.ping():
                self._redis = None
                self._get_from_redis = False
        else:
            self._redis = None

    def process_patients(self, patients):
        return patients

    def get_standardization_values(self, collection: str) -> tp.Dict[str, tp.Dict[str, float]]:
        logger.info('Getting standardization values')
        db = init_database(config_name=self.config_name)
        return next(db[collection].aggregate([
            {
                '$group': {
                    '_id': '$name',
                    'avg': {
                        '$avg': '$value'
                    },
                    'std': {
                        '$stdDevPop': '$value'
                    }
                }
            }, {
                '$addFields': {
                    'dummy': 1
                }
            }, {
                '$group': {
                    '_id': '$dummy',
                    'data': {
                        '$push': {
                            'k': '$_id',
                            'v': {
                                'avg': '$avg',
                                'std': '$std'
                            }
                        }
                    }
                }
            }, {
                '$project': {
                    'data': {
                        '$arrayToObject': '$data'
                    },
                    '_id': 0
                }
            }, {
                '$replaceRoot': {
                    'newRoot': {
                        '$mergeObjects': [
                            '$$ROOT', '$data'
                        ]
                    }
                }
            }, {
                '$project': {
                    'data': 0
                }
            }
        ]))

    def _get_patient_samples_dict(self, patients: tp.List[str], collection: str):
        logger.info('Getting patient`s samples')
        db = init_database(config_name=self.config_name)
        items: dict = next(db[collection].aggregate([
            {
                '$group': {
                    '_id': '$patient',
                    'samples': {
                        '$addToSet': '$sample'
                    }
                }
            }, {
                '$project': {
                    'patient': '$_id',
                    '_id': 0,
                    'samples': 1,
                    'dummy': '1'
                }
            }, {
                '$group': {
                    '_id': '$dummy',
                    'data': {
                        '$push': {
                            'k': '$patient',
                            'v': '$samples'
                        }
                    }
                }
            }, {
                '$project': {
                    'data': {
                        '$arrayToObject': '$data'
                    }
                }
            }, {
                '$replaceRoot': {
                    'newRoot': {
                        '$mergeObjects': [
                            '$$ROOT', '$data'
                        ]
                    }
                }
            }, {
                '$project': {
                    '_id': 0,
                    'data': 0
                }
            }
        ], allowDiskUse=True
        )
        )
        return {key: value for key, value in items.items() if key in patients}

    def _get_raw_attributes(self, collection: str, sample: str):
        if self._get_from_redis:
            redis_id = f'{sample}-{collection}'
            attributes = self._redis.get(redis_id)
            if attributes is not None:
                # logger.debug('Found in redis')
                return json.loads(attributes)
        logger.debug(f'Not found in redis {sample}-{collection}')
        db = init_database(config_name=self.config_name)
        return next(db[collection].aggregate([
            {
                '$match': {
                    'sample': sample
                }
            }, {
                '$group': {
                    '_id': '$sample',
                    'data': {
                        '$push': {
                            'k': '$name',
                            'v': '$value'
                        }
                    },
                    'patient': {
                        '$push': '$patient'
                    }
                }
            }, {
                '$project': {
                    'user': {
                        '$arrayElemAt': [
                            '$patient', 0
                        ]
                    },
                    'sample': '$_id',
                    '_id': 0,
                    'data': {
                        '$arrayToObject': '$data'
                    }
                }
            }, {
                '$replaceRoot': {
                    'newRoot': {
                        '$mergeObjects': [
                            '$$ROOT', '$data'
                        ]
                    }
                }
            }, {
                '$project': {
                    'data': 0,
                    'user': 0,
                    'sample': 0
                }
            }
        ]))

    def _get_collection_attributes(self, collection: str):
        db = init_database(config_name=self.config_name)
        return db[collection].distinct('name')

    @abstractmethod
    def get_attributes(self, sample, collection_name: str):
        ...

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_db'] = None
        state['_collection'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @abstractmethod
    def define_samples(self) -> tp.List[tp.Any]:
        ...

    def get_sample(self, sample: str) -> dict:
        ...

    @classmethod
    def get_standardization_dict(cls, collection, patients: tp.List[str], config_name: tp.Optional[str] = None) -> \
            tp.Dict[str, tp.Dict[str, float]]:
        logger.info('Getting standardization values')
        db = init_database(config_name=config_name)
        return next(db[collection].aggregate(
            [
                {
                    '$group': {
                        '_id': '$name',
                        'avg': {
                            '$avg': '$value'
                        },
                        'std': {
                            '$stdDevPop': '$value'
                        }
                    }
                }, {
                '$addFields': {
                    'dummy': 1
                }
            }, {
                '$group': {
                    '_id': '$dummy',
                    'data': {
                        '$push': {
                            'k': '$_id',
                            'v': {
                                'avg': '$avg',
                                'std': '$std'
                            }
                        }
                    }
                }
            }, {
                '$project': {
                    'data': {
                        '$arrayToObject': '$data'
                    },
                    '_id': 0
                }
            }, {
                '$replaceRoot': {
                    'newRoot': {
                        '$mergeObjects': [
                            '$$ROOT', '$data'
                        ]
                    }
                }
            }, {
                '$project': {
                    'data': 0
                }
            },

            ], allowDiskUse=True))

    def __len__(self):
        return len(self.samples)


class AttributeFillerDataset(BaseDataset, Dataset):

    def __init__(
            self,
            patients: tp.List[str],
            collection_name: str,
            attributes_drop_rate: float = 0.05,
            config_name: str = 'omicsdb',
            raw_attributes_file: str = Path(tempfile.gettempdir(), 'raw_attributes_file.json'),
            override_raw_attributes_file: bool = True
    ):
        self._collection_name = collection_name
        self._attributes_drop_rate = attributes_drop_rate

        super(AttributeFillerDataset, self).__init__(patients=patients, config_name=config_name)

        db = init_database(config_name=self.config_name)
        self._attributes = db[collection_name].distinct('name')

        self._all_raw_attributes = self._get_all_raw_attributes(patients=self.patients)

    def get_attributes(self, sample, collection_name: str):
        # raw_attributes_dict = self._get_raw_attributes(sample=sample, collection=collection_name)
        raw_attributes_dict: dict = [item for item in self._all_raw_attributes if item['sample'] == sample][0].copy()
        raw_attributes_dict.pop('sample')

        attributes_vec = np.zeros(len(self._get_collection_attributes(collection=collection_name)), dtype=np.float32)

        for i, attribute in enumerate(self._attributes):
            if attribute in raw_attributes_dict.keys():
                if not math.isnan(raw_attributes_dict[attribute]):
                    attributes_vec[i] = raw_attributes_dict[attribute]
                else:
                    attributes_vec[i] = 0

        return attributes_vec

    def get_sample(self, sample: str) -> dict:
        attributes_vec = self.get_attributes(sample=sample, collection_name=self._collection_name)
        targets = attributes_vec.copy()
        drop_indices = random.sample(list(np.arange(0, len(self._attributes))),
                                     int(self._attributes_drop_rate * len(self._attributes)))

        for drop_ind in drop_indices:
            attributes_vec[drop_ind] = 0.

        return dict(
            targets=targets,
            attributes=attributes_vec,
            dropped_attributes_index=np.array(drop_indices, dtype=np.int16),
            dropped_attributes=[self._attributes[i] for i in drop_indices],
            attributes_names=self._attributes
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.get_sample(self.samples[item][-1])

    def define_samples(self):
        patient_samples_dict = self._get_patient_samples_dict(patients=self.patients, collection=self._collection_name)
        return [(patient, sample) for patient in patient_samples_dict.keys() for sample in
                patient_samples_dict[patient]]

    @staticmethod
    def get_raw_attributes_query(patients: tp.List[str] = None):
        ppln = [
            {
                '$group': {
                    '_id': '$sample',
                    'data': {
                        '$push': {
                            'k': '$name',
                            'v': '$value'
                        }
                    },
                    'patient': {
                        '$push': '$patient'
                    }
                }
            }, {
                '$project': {
                    'user': {
                        '$arrayElemAt': [
                            '$patient', 0
                        ]
                    },
                    'sample': '$_id',
                    '_id': 0,
                    'data': {
                        '$arrayToObject': '$data'
                    }
                }
            }, {
                '$replaceRoot': {
                    'newRoot': {
                        '$mergeObjects': [
                            '$$ROOT', '$data'
                        ]
                    }
                }
            }, {
                '$project': {
                    'data': 0,
                    'user': 0
                }
            }
        ]

        if patients is not None:
            ppln.insert(
                0,
                {
                    '$match': {
                        'patient': {
                            '$in': patients
                        }
                    }
                }

            )
        return ppln

    @classmethod
    def dump_raw_attributes_file(cls, output_file: str, collection: str, config_name: str,
                                 patients: tp.List[str] = None):
        db = init_database(config_name=config_name)
        items = list(db[collection].aggregate(cls.get_raw_attributes_query(patients), allowDiskUse=True))
        with open(output_file, 'w') as f:
            json.dump(items, f, indent=2)

    def _get_all_raw_attributes(
            self,
            patients: tp.List[str] = None
    ):

        db = init_database(config_name=self.config_name)
        items = list(db[self._collection_name].aggregate(self.get_raw_attributes_query(patients), allowDiskUse=True))
        return items


class AttentionMixin(AttributeFillerDataset):
    def get_sample(self, sample: str) -> dict:
        sample = super().get_sample(sample)
        attention = np.ones_like(sample['attributes'])
        for index in sample['dropped_attributes_index']:
            attention[index] = 0.

        sample['attributes'] = np.stack((sample['attributes'], attention))
        return sample


class MultiOmicsDataset(BaseDataset, Dataset):
    def __init__(self, patients: tp.List[str],
                 collections: tp.List[str],
                 config_name: str = 'brca-reader',
                 get_from_redis: bool = False):
        self._collections = collections

        super().__init__(patients, config_name=config_name, get_from_redis=get_from_redis)

        db = init_database(config_name=config_name)
        self.attributes_order = {collection_name: sorted(db[collection_name].distinct('name')) for collection_name in
                                 self._collections}

        self.patient_names_col_samples = {
            col: self.get_all_patients_samples_in_collection(collection=col) for col in self._collections
        }

        self.modality_patients = {modality: self._get_patients_in_modality(modality) for modality in self._collections}

    def get_all_patients_samples_in_collection(self, collection: str):
        db = init_database(self.config_name)

        res = {item['patient']: item['samples'] for item in (db[collection].aggregate([
            {'$match': {'patient': {'$in': self.patients}}},
            {
                '$group': {
                    '_id': '$patient',
                    'samples': {
                        '$push': '$sample'
                    }
                }
            }, {
                '$project': {
                    'patient': '$_id',
                    '_id': 0,
                    'samples': 1
                }
            }
        ], allowDiskUse=True)
        )}
        cache[hash(f'get_all_patients_samples_in_collection-{collection}')] = res
        return res

    def _get_patient_samples_dict(self, patients: tp.List[str], collection: str) -> dict:
        patient_samples = dict()
        db = init_database(self.config_name)
        for patient in patients:
            patient_samples[patient] = db[collection].find(dict(patient=patient)).distinct('sample')

        return patient_samples

    def define_samples(self) -> tp.List[tp.Any]:

        samples = []
        for patient in self.patients:
            patients: tp.List = self.patients.copy()
            if patient in patients:
                patients.remove(patient)
            for patient_ in patients:
                samples.append((patient, patient_))

        return samples

    def _get_patients_in_modality(self, collection: str):
        db = init_database(self.config_name)
        return db[collection].find({'patient': {'$in': self.patients}}).distinct('patient')

    def process_patients(self, patients: tp.List):
        patients_dict = dict()

        db = init_database(self.config_name)
        for col in self._collections:
            patients_dict[col] = list(set(db[col].distinct('patient')).intersection(patients))

        self.patients_dict_by_col = patients_dict
        return patients

    def get_patients(self):
        db = init_database(self.config_name)
        for col in self._collections:
            db[col].distinct('patient')

    def get_sample_names_for_patient(self, collection: str, patient: str):
        db = init_database(self.config_name)

        return db[collection].find({'patient': patient}).distinct('sample')

    def get_base_patient(self, anchor_collection: str, pos_collection: str) -> str:
        """
        Returns the base patient to be used as anchor and positive sample. This patient must exist in both the anchor
        and the positive modality so it must be selected from the subset of patients existing in boths

        :param anchor_collection: name of the anchor collection being used
        :param pos_collection: name of the positive collection being used
        :return: name of the selected patient
        """
        base_patient = random.choice(
            list(
                set(
                    self.modality_patients[anchor_collection]).intersection(
                    set(
                        self.modality_patients[pos_collection]
                    )
                )
            )
        )
        return base_patient

    def get_anchor_sample(self, patient: str, collection: str) -> str:
        """
        Returns the name of the sample to be used as anchor from the anchor collection
        :param patient: name of the patient being used. The patient must exist in the requested collection
        :param collection: name of the anchor collection
        :return:
        """
        anchor_sample = random.choice(
            self.patient_names_col_samples[collection][patient]
        )
        return anchor_sample

    def get_pos_sample(self, patient: str, collection: str) -> str:
        """
        Returns the name of the sample to be used as positive from the positive collection
        :param patient: name of the patient being used. The patient must exist in the requested collection
        :param collection: name of the positive collection
        :return:
        """
        pos_sample = random.choice(
            self.patient_names_col_samples[collection][patient]
        )
        return pos_sample

    def get_neg_sample(self, patient, collection: str) -> tp.Tuple[str, str]:
        """
        Returns the name of the sample to be used as negative from the negative collection. The patient to be used is
        selected randomally according to the available patients in the collection but excluding the anchor/positive
        patient

        :param anchor_patient: id of the patient being used as anchor. This patient must not be selected
        :param collection: name of the positive collection
        :return:
        """

        neg_sample = random.choice(
            self.patient_names_col_samples[collection][patient]
        )
        return neg_sample

    def get_survival_time(self, patient):
        db = init_database(config_name=self.config_name)
        patient_meta_survival = db['Survival'].find_one(dict(patient=patient, name='OS.time'))
        if not patient_meta_survival or math.isnan(patient_meta_survival['value']):
            return np.inf
        return patient_meta_survival['value']

    def get_neg_patient(self, anchor_patient: str, collection: str):

        available_patients: tp.List = self.modality_patients[collection].copy()
        if anchor_patient in available_patients:
            available_patients.remove(anchor_patient)
        neg_patient = random.choice(self.modality_patients[collection])
        return neg_patient

    def get_sample(self,
                   anchor_collection: str,
                   pos_collection: str,
                   neg_collection: str
                   ) -> dict:

        base_patient = self.get_base_patient(anchor_collection=anchor_collection, pos_collection=pos_collection)
        neg_patient = self.get_neg_patient(anchor_patient=base_patient, collection=neg_collection)

        anchor_sample = self.get_anchor_sample(patient=base_patient, collection=anchor_collection)
        pos_sample = self.get_pos_sample(patient=base_patient, collection=pos_collection)
        neg_sample = self.get_neg_sample(patient=neg_patient, collection=neg_collection)

        anchor_attributes = self.get_attributes(sample=anchor_sample, collection_name=anchor_collection)

        pos_attributes = self.get_attributes(sample=pos_sample, collection_name=pos_collection)

        neg_attributes = self.get_attributes(sample=neg_sample, collection_name=neg_collection)

        return dict(
            anchor=anchor_attributes,
            pos=pos_attributes,
            neg=neg_attributes,
            anchor_survival=self.get_survival_time(base_patient),
            pos_survival=self.get_survival_time(base_patient),
            neg_survival=self.get_survival_time(neg_patient)
        )

    def get_attributes(self, sample, collection_name: str):
        raw_attributes_dict = self._get_raw_attributes(collection=collection_name, sample=sample)

        attributes = self.attributes_order[collection_name]
        attributes_vec = np.zeros(len(attributes), dtype=np.float32)

        # attributes = db[collection_name].distinct('name').sort()

        for i, attribute in enumerate(attributes):
            if attribute in raw_attributes_dict.keys():
                if not math.isnan(raw_attributes_dict[attribute]):
                    attributes_vec[i] = raw_attributes_dict[attribute]
                else:
                    attributes_vec[i] = 0.
            else:
                attributes_vec[i] = 0.

        return attributes_vec

    def get_samples(self, anchor_collection, pos_collection, neg_collection, num_samples: int):

        return [self.get_sample(anchor_collection=anchor_collection,
                                pos_collection=pos_collection,
                                neg_collection=neg_collection)
                for i in range(num_samples)
                ]

    def __getitem__(self, item):
        anchor_collection = random.choice(self._collections)
        collections = self._collections.copy()
        collections.remove(anchor_collection)
        pos_collection = random.choice(collections)
        neg_collection = random.choice(self._collections)

        items = self.get_samples(
            anchor_collection=anchor_collection,
            pos_collection=pos_collection,
            neg_collection=neg_collection,
            num_samples=len(item)
        )

        return dict(
            anchor=torch.from_numpy(np.stack([item['anchor'] for item in items])),
            pos=torch.from_numpy(np.stack([item['pos'] for item in items])),
            neg=torch.from_numpy(np.stack([item['neg'] for item in items])),
            anchor_survival=torch.from_numpy(np.stack([item['anchor_survival'] for item in items])),
            pos_survival=torch.from_numpy(np.stack([item['pos_survival'] for item in items])),
            neg_survival=torch.from_numpy(np.stack([item['neg_survival'] for item in items])),
            anchor_modality=anchor_collection,
            pos_modality=pos_collection,
            neg_modality=neg_collection
        )

    def __len__(self):
        return len(self.samples)


class AttributeSignDataset(AttributeFillerDataset):
    def get_sample(self, sample: str) -> dict:
        sample = super().get_sample(sample)

        targets = sample['targets']
        targets_ = np.zeros_like(targets)

        for index in sample['dropped_attributes_index']:
            targets_[index] = np.sign(targets[index])

        sample['target_attributes'] = targets
        sample['targets'] = targets_
        return sample
