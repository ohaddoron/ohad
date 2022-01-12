import random
import typing as tp

import numpy as np
from torch.utils.data import Dataset

from common.database import init_database


class AttributeFillerDataset(Dataset):
    def __init__(self, patients: tp.List[str], collection: str, attributes_drop_rate: float = 0.2, standardize=True):

        self._standardize = standardize
        self._collection_name = collection
        self._init_db()
        self._attributes_drop_rate = attributes_drop_rate
        self._attributes = self._collection.distinct('name')
        self._patient_samples_dict = self._get_patient_samples_dict()
        self._standardization_dict = self._get_standardization_values()

        self._samples = [(patient, sample) for patient in self._patient_samples_dict.keys() for sample in
                         self._patient_samples_dict[patient]]

        self.patients = patients

    def _init_db(self):
        self._db = init_database(config_name='brca-reader')
        self._collection = self._db[self._collection_name]

    def _get_standardization_values(self):
        return next(self._collection.aggregate([
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

    def _get_sample(self, sample):
        attributes_vec = self._get_attributes(sample=sample)
        targets = attributes_vec.copy()
        drop_indices = random.sample(list(np.arange(0, len(self._attributes))),
                                     int(self._attributes_drop_rate * len(self._attributes)))

        for drop_ind in drop_indices:
            attributes_vec[drop_ind] = 0.

        return dict(
            targets=targets,
            attributes=attributes_vec,
            dropped_attributes_index=np.array(drop_indices, dtype=np.int16),
            dropped_attributes=[self._attributes[i] for i in drop_indices])

    def _get_attributes(self, sample):
        raw_attributes_dict = self._get_raw_attributes(sample)

        attributes_vec = np.zeros(len(self._attributes), dtype=np.float32)

        for i, attribute in enumerate(self._attributes):
            if attribute in raw_attributes_dict.keys():
                attributes_vec[i] = raw_attributes_dict[attribute]
                if self._standardize:
                    attributes_vec[i] = (attributes_vec[i] - self._standardization_dict[attribute]['avg']) / \
                                        (self._standardization_dict[attribute]['std'] + np.finfo(np.float32).eps)

        return attributes_vec

    def _get_raw_attributes(self, sample):
        return next(self._collection.aggregate([
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

    def _get_patient_samples_dict(self):
        return next(self._collection.aggregate([
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

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, item):
        return self._get_sample(self._samples[item][-1])

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_db'] = None
        state['_collection'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_db()
