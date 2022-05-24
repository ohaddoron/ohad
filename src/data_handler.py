import csv
import hashlib
import json
from abc import ABC
from pathlib import Path
from typing import List, Dict

import typer
from loguru import logger
import numpy as np
import redis
import pymongo
from typer import Typer

app = Typer(name='Data Handler')


class AbstractDataHandler(ABC):
    def __init__(self, redis_host: str = None, mongodb_connection_string: str = None, db_name: str = None,
                 cache: bool = False,
                 force_cache: bool = False,
                 redis_port: int = 6379):
        self.force_cache = force_cache
        if redis_host is None and mongodb_connection_string is None:
            raise ValueError('Must provide either redis connection string or mongo connection string, or both but not '
                             'neither')

        if mongodb_connection_string is not None and db_name is None:
            raise ValueError('Cannot provide mongodb connection string and not provide database name')

        if redis_host:
            self._redis = redis.Redis(host=redis_host, port=redis_port)
        else:
            self._redis = dict()

        self.cache = cache if self._redis else False  # allows caching if redis is available

        if mongodb_connection_string:
            client = pymongo.MongoClient(mongodb_connection_string)
            logger.debug(client.server_info())
            self._db = client[db_name]
            logger.debug(self)
        else:
            self._db = None


class PatientDataHandler(AbstractDataHandler):

    def fetch_patient_specific_data(self, patient: str, modality: str, features: List[str]):
        logger.debug(f'Fetching patient info for {features}')
        hashes = {
            feature: self.generate_patient_modality_feature_hash(patient=patient, modality=modality, feature=feature)
            for feature in
            features}

        results = self.get_patient_info_from_redis(hashes)

        if not any([item['value'] for item in results]):
            logger.debug(f'Some features were missing in Redis for patient {patient}, looking up in MongoDB')
            results = self.get_patient_info_from_mongo(patient=patient, modality=modality, features=features)
            logger.debug(f'MongoDB query for patient {patient} completed')
            if self.cache:
                logger.debug(f'Filling in missing information in Redis for patient: {patient}')
                self.set_patient_specific_data(patient=patient,
                                               modality=modality,
                                               features={item['feature']: item['value'] for item in results}
                                               )

        return results

    def get_patient_info_from_redis(self, hashes: Dict[str, hash]):
        if not self.force_cache:
            results = [dict(feature=feature, value=self._redis.get(hash_)) for feature, hash_ in hashes.items()]
        else:
            results = [dict(feature='dummy', value=None)]
        return results

    def get_patient_info_from_mongo(self, patient: str, modality: str, features: List[str]):
        return list(
            self._db[modality].aggregate(
                [
                    {
                        '$match': {
                            'patient': patient
                        }
                    }, {
                    '$match': {
                        'name': {
                            '$in': features
                        }
                    }
                }, {
                    '$project': {
                        'feature': '$name',
                        'value': '$value',
                        '_id': 0
                    }
                }
                ]
            )
        )

    def set_patient_specific_data(self, patient: str, modality: str, features: Dict[str, float]):
        """
        Caches patient specfic data to redis
        :param patient: patient ID
        :type patient: str
        :param modality: modality being used, e.g. ['mRNA', 'DNAm',...]
        :type modality: str
        :param features: Key value pairs of features and values
        :type features: Dict[str, float]
        """
        for feature, value in features.items():
            key = self.generate_patient_modality_feature_hash(patient=patient, modality=modality, feature=feature)
            self._redis[key] = value

    @staticmethod
    def generate_patient_modality_feature_hash(patient: str, modality: str, feature: str) -> str:
        return f'{patient}-{modality}-{feature}'


class FeatureDataHandler(AbstractDataHandler):
    @staticmethod
    def generate_feature_modality_hash(modality: str, feature: str) -> str:
        return f'{modality}-{feature}'

    def get_feature_info_from_redis(self, hsh):
        if self.force_cache:
            return None
        else:
            return self._redis.get(hsh)

    def get_feature_info_from_mongo(self, modality: str, feature: str) -> np.ndarray:
        return np.array(
            [item['value'] for item in self._db[modality].aggregate(
                [
                    {
                        '$match': {
                            'name': feature
                        }
                    }, {
                    '$project': {
                        'feature': '$name',
                        'value': '$value',
                        '_id': 0
                    }
                }
                ]
            )
             ]
        )

    def fetch_feature_specific_data(self, feature: str, modality: str):
        hsh = self.generate_feature_modality_hash(modality=modality, feature=feature)

        result = self.get_feature_info_from_redis(hsh)

        if result is None:
            result = self.get_feature_info_from_mongo(modality=modality, feature=feature)
            if self.cache:
                self.set_feature_specific_data(
                    feature=feature,
                    modality=modality,
                    value=result
                )

        return result

    def set_feature_specific_data(self, modality: str, feature: str, value: np.ndarray):
        self._redis[self.generate_feature_modality_hash(modality=modality, feature=feature)] = value


class FeaturesMetadataHandler(AbstractDataHandler):
    def fetch_features_metadata(self, modality: str, features: List[str], patients: List[str] = None):
        results: list = self.get_features_metadata_from_redis(modality, features)
        missing_features = [result['feature'] for result in results if result['metadata'] is None]
        if not missing_features:
            return results

        results = [result for result in results if result['metadata'] is not None]

        missing_data = self.get_features_metadata_from_mongo(modality=modality,
                                                             features=missing_features,
                                                             patients=patients)
        if self.cache:
            [self.set_feature_metadata(modality=modality, feature=data['feature'], value=data['metadata']) for data in
             missing_data]
        results += missing_data
        return results

    def set_feature_metadata(self, modality: str, feature: str, value: dict):
        self._redis[self.generate_feature_modality_metadata_hash(modality=modality, feature=feature)] = json.dumps(
            value)

    def get_features_metadata_from_redis(self, modality, features: List[str]) -> List[dict]:
        if self.force_cache:
            return [dict(feature=feature, metadata=None) for feature in features]
        features_metadata = []
        for feature in features:
            metadata = self._redis.get(
                self.generate_feature_modality_metadata_hash(
                    modality=modality,
                    feature=feature
                )
            )
            if metadata:
                metadata = json.loads(metadata)
            features_metadata.append(
                dict(
                    feature=feature,
                    metadata=metadata
                )
            )
        return features_metadata

    def get_features_metadata_from_mongo(self, modality: str, features: List[str], patients: List[str] = None):
        if features is None:
            features = list(self._db[modality].distinct('name'))
        return [dict(feature=item['feature'], metadata=item['metadata']) for item in list(self._db[modality].aggregate([
            {
                '$match': {
                    'name': {
                        '$in':
                            features
                    },
                    'patient': {
                        '$in':
                            patients
                    }
                } if patients else {'name': {
                    '$in':
                        features
                }, 'patient': {'$ne': None}}
            }, self.get_metadata_aggregation_values(), {
                '$addFields': {
                    'var': {
                        '$pow': [
                            '$stdev', 2
                        ]
                    }
                }
            }, {
                '$project': {
                    'metadata': {
                        'max': '$max',
                        'min': '$min',
                        'stdev': '$stdev',
                        'avg': '$avg',
                        'var': {
                            '$pow': [
                                '$stdev', 2
                            ]
                        }
                    },
                    'feature': '$_id',
                    '_id': 0
                }
            }
        ]))]

    @staticmethod
    def generate_feature_modality_metadata_hash(modality: str, feature: str) -> str:
        return f'{modality}-{feature}-metadata'

    def get_metadata_aggregation_values(self):
        return {
            '$group': {
                '_id': '$name',
                'max': {
                    '$max': '$value'
                },
                'min': {
                    '$min': '$value'
                },
                'stdev': {
                    '$stdDevPop': '$value'
                },
                'avg': {
                    '$avg': '$value'
                }
            }
        }


@app.command()
def generate_metadata(
        mongodb_connection_string: str = typer.Option(..., help='MongoDB connection string to draw data from',
                                                      prompt_required=True),
        db_name: str = typer.Option(..., help='Database name', prompt_required=True),
        modality: str = typer.Option(..., help='Modality to compute metadata for'),
        redis_host: str = typer.Option(...,
                                       help='Redis connection string. Metadata will be automatically '
                                            'inserted'),
        patients_file: Path = typer.Option(None,
                                           help='Path to a csv file containing patients in comma seperated values for '
                                                'patients to be used when computing metadata. If not provided, '
                                                'will compute for all patients',
                                           exists=True, file_okay=True, readable=True, resolve_path=True),
        features_file: Path = typer.Option(None,
                                           help='Path to a csv file containing features in comma seperated values for '
                                                'patients to be used when computing metadata. If not provided, '
                                                'will compute for all features',
                                           exists=True, file_okay=True, readable=True, resolve_path=True),
        force_override: bool = typer.Option(False, help='If True, will override the data already existing in Redis')
):
    if patients_file:
        patients = list(csv.reader(patients_file.open()))[0]
    else:
        patients = None
    if features_file:
        features = list(csv.reader(features_file.open()))[0]
    else:
        features = None
    return FeaturesMetadataHandler(mongodb_connection_string=mongodb_connection_string,
                                   db_name=db_name,
                                   redis_host=redis_host,
                                   force_cache=force_override,
                                   cache=True
                                   ).fetch_features_metadata(patients=patients, features=features, modality=modality)


if __name__ == '__main__':
    app()
