import json
import math
import random
import typing as tp
from abc import abstractmethod
from functools import lru_cache
from pathlib import Path

import diskcache
import numpy as np
from torch.utils.data import Dataset

from common.database import init_database
from src.logger import logger
from src.cache import cache

cache: diskcache.Cache


class BaseDataset(Dataset):
    def __init__(self, patients: tp.List[str], config_name: str = 'omicsdb'):
        self.patients = patients
        self.config_name = config_name
        self.samples = self.define_samples()

    def get_standardization_values(self, collection: str) -> tp.Dict[str, tp.Dict[str, float]]:
        logger.info('Getting standardization values')
        db = init_database(config_name=self._config_name)
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
        self.init_db()

    @abstractmethod
    def define_samples(self) -> tp.List[tp.Any]:
        ...

    def get_sample(self, sample: str) -> dict:
        ...

    def get_standardization_dict(self, collection, patients: tp.List[str]) -> tp.Dict[str, tp.Dict[str, float]]:
        logger.info('Getting standardization values')
        logger.warning(
            'Ignoring patient names when fetching standardization values - this should be transferred elsewhere')
        db = init_database(config_name=self.config_name)
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
            }
            ], allowDiskUse=True))

    def __len__(self):
        return len(self.samples)


class AttributeFillerDataset(BaseDataset):

    def __init__(self, patients: tp.List[str],
                 collection_name: str,
                 attributes_drop_rate: float = 0.05,
                 config_name: str = 'omicsdb'):
        self._collection_name = collection_name
        self._attributes_drop_rate = attributes_drop_rate

        super(AttributeFillerDataset, self).__init__(patients=patients, config_name=config_name)
        # self._standardization_dict = self.get_standardization_dict(collection=collection_name, patients=patients)

        db = init_database(config_name=self.config_name)
        self._attributes = db[collection_name].distinct('name')

        self._all_raw_attributes = self._get_all_raw_attributes()

    def init_db(self):
        super().init_db()

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
            dropped_attributes=[self._attributes[i] for i in drop_indices]
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
    def get_raw_attributes_query():
        return [
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

    @classmethod
    def dump_raw_attributes_file(cls, output_file: str, collection: str, config_name: str):
        db = init_database(config_name=config_name)
        items = list(db[collection].aggregate(cls.get_raw_attributes_query(), allowDiskUse=True))
        with open(output_file, 'w') as f:
            json.dump(items, f, indent=2)

    def _get_all_raw_attributes(self):
        raw_attributes_file = Path(Path(__file__).parent, '../resources/gene_expression_attributes.json')
        if raw_attributes_file.exists():
            return json.load(raw_attributes_file.open('r'))
        db = init_database(config_name=self.config_name)
        items = list(db[self._collection_name].aggregate(self.get_raw_attributes_query(), allowDiskUse=True))
        json.dump(items, raw_attributes_file.open('w'), indent=2)
        return items


class MultiOmicsDataset(BaseDataset):
    def __init__(self, patients: tp.List[str], collections: tp.List[str]):
        self._collections = collections

        super().__init__(patients)

        self.standardization_dicts = self.get_standardization_dict_multiple_collections(collections=collections,
                                                                                        patients=patients)

    def get_standardization_dict_multiple_collections(self, collections: tp.List[str], patients: tp.List[str]) -> \
            tp.Dict[str, tp.Dict[str, tp.Dict[str, float]]]:
        return {col: self.get_standardization_dict(collection=col, patients=patients) for col in collections}

    def define_samples(self) -> tp.List[tp.Any]:
        patients_samples_dict = {col: self._get_patient_samples_dict(patients=self.patients, collection=col) for col in
                                 self._collections}
        samples = []
        for col in self._collections[:-1]:
            for col_ in self._collections[1:]:
                for patient in self.patients:
                    _patients = self.patients.copy()
                    _patients.remove(patient)

                    r_col = random.choice(self._collections)
                    samples.append(
                        (
                            (random.choice(patients_samples_dict[col][patient]), col),
                            (random.choice(patients_samples_dict[col_][patient]), col_),
                            (
                                random.choice(patients_samples_dict[r_col][random.choice(_patients)]),
                                r_col
                            )
                        )
                    )

        return samples
