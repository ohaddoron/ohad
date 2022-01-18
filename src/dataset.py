import random
import typing as tp
from abc import abstractmethod

import numpy as np
from torch.utils.data import Dataset

from common.database import init_database


class BaseDataset(Dataset):
    def __init__(self, patients: tp.List[str]):
        self.patients = patients
        self.init_db()
        self._samples = self.define_samples()
        self.define_samples()

    def init_db(self):
        self._db = init_database(config_name='brca-reader')

    def get_standardization_values(self, collection: str) -> tp.Dict[str, tp.Dict[str, float]]:
        return next(self._db[collection].aggregate([
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

    def _get_patient_samples_dict(self, patients, collection):
        return next(self._db[collection].aggregate([
            {
                '$match': {
                    'patient': {
                        '$in': patients
                    }
                }
            }, {
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

    def _get_raw_attributes(self, collection: str, sample: str):
        return next(self._db[collection].aggregate([
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
        return self._db[collection].distinct('name')

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

    def get_standardization_dict(self, collection, patients: tp.List[str]):
        return next(self._db[collection].aggregate(
            [
                {
                    '$match': {
                        'patient': {
                            '$in': patients
                        }
                    }
                }, {
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
        return len(self._samples)


class AttributeFillerDataset(BaseDataset):

    def __init__(self, patients: tp.List[str],
                 collection_name: str,
                 standardize: bool = True,
                 attributes_drop_rate: float = 0.05):
        self._collection_name = collection_name
        self._standardize = standardize
        self._attributes_drop_rate = attributes_drop_rate

        super(AttributeFillerDataset, self).__init__(patients=patients)
        self._standardization_dict = self.get_standardization_dict(collection=collection_name, patients=patients)

        self._attributes = self._db[collection_name].distinct('name')

    def init_db(self):
        super().init_db()

    def get_attributes(self, sample, collection_name: str):
        raw_attributes_dict = self._get_raw_attributes(sample=sample, collection=collection_name)

        attributes_vec = np.zeros(len(self._get_collection_attributes(collection=collection_name)), dtype=np.float32)

        for i, attribute in enumerate(self._get_raw_attributes(collection=collection_name, sample=sample)):
            if attribute in raw_attributes_dict.keys():
                attributes_vec[i] = raw_attributes_dict[attribute]
                if self._standardize:
                    attributes_vec[i] = (attributes_vec[i] - self._standardization_dict[attribute]['avg']) / \
                                        (self._standardization_dict[attribute]['std'] + np.finfo(np.float32).eps)

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
        return len(self._samples)

    def __getitem__(self, item):
        return self.get_sample(self._samples[item][-1])

    def define_samples(self):
        patient_samples_dict = self._get_patient_samples_dict(patients=self.patients, collection=self._collection_name)
        return [(patient, sample) for patient in patient_samples_dict.keys() for sample in
                patient_samples_dict[patient]]


class MultiOmicsDataset(BaseDataset):
    def __init__(self, patients: tp.List[str], collections: tp.List[str]):
        super().__init__(patients)
        