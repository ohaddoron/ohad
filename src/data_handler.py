import hashlib
from typing import List, Dict

import redis
import pymongo


class DataHandler:
    def __init__(self, redis_connection_string: str = None, mongodb_connection_string: str = None, db_name: str = None,
                 cache: bool = False,
                 force_cache: bool = False):
        self.force_cache = force_cache
        if redis_connection_string is None and mongodb_connection_string is None:
            raise ValueError('Must provide either redis connection string or mongo connection string, or both but not '
                             'neither')

        if mongodb_connection_string is not None and db_name is None:
            raise ValueError('Cannot provide mongodb connection string and not provide database name')

        if redis_connection_string:
            self._redis = redis.Redis(redis_connection_string)
        else:
            self._redis = dict()

        self.cache = cache if self._redis else False  # allows caching if redis is available

        if mongodb_connection_string:
            self._db = pymongo.MongoClient(mongodb_connection_string)[db_name]
        else:
            self._db = None

    def fetch_patient_specific_data(self, patient: str, modality: str, features: List[str]):
        hashes = {
            feature: self.generate_patient_modality_feature_hash(patient=patient, modality=modality, feature=feature)
            for feature in
            features}

        results = self.get_patient_info_from_redis(hashes)

        if not any([item['value'] for item in results]):
            results = self.get_patient_info_from_mongo(patient=patient, modality=modality, features=features)
            if self.cache:
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

    def generate_feature_modality_hash(self, modality: str, feature: str) -> str:
        return f'{modality}-{feature}'

    def get_feature_specific_data(self, feature: str, modality: str):
        hash = self.generate_feature_modality_hash(modality=modality, feature=feature)

        results = self.get_patient_info_from_redis(hashes)

        if not any([item['value'] for item in results]):
            results = self.get_patient_info_from_mongo(patient=patient, modality=modality, features=features)
            if self.cache:
                self.set_patient_specific_data(patient=patient,
                                               modality=modality,
                                               features={item['feature']: item['value'] for item in results}
                                               )

        return results
