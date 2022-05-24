import csv
from tempfile import TemporaryDirectory

import mongomock
import numpy as np
import pymongo
import pytest
from pathlib import Path
import redis
from bson import json_util

from src.data_handler import PatientDataHandler, FeatureDataHandler, FeaturesMetadataHandler, generate_metadata
from unittest.mock import patch, MagicMock


def insert_data_to_database():
    items = json_util.loads(Path(__file__).parent.joinpath('../resources/test_data_handler/mRNA.json').open().read())
    with pymongo.MongoClient('mongodb://localhost:27017') as client:
        client['TCGAOmics']['mRNA'].insert_many(items)


def get_patient_data_handler(cache: bool = False, force_cache: bool = False) -> PatientDataHandler:
    return PatientDataHandler(mongodb_connection_string='mongodb://localhost:27017',
                              db_name='TCGAOmics',
                              redis_connection_string='redis://localhost:6379',
                              cache=cache,
                              force_cache=force_cache)


def get_feature_data_handler(cache: bool = False, force_cache: bool = False) -> FeatureDataHandler:
    return FeatureDataHandler(mongodb_connection_string='mongodb://localhost:27017',
                              db_name='TCGAOmics',
                              redis_connection_string='redis://localhost:6379',
                              cache=cache,
                              force_cache=force_cache)


def get_features_metadata_handler(cache: bool = False, force_cache: bool = False) -> FeaturesMetadataHandler:
    return FeaturesMetadataHandler(mongodb_connection_string='mongodb://localhost:27017',
                                   db_name='TCGAOmics',
                                   redis_connection_string='redis://localhost:6379',
                                   cache=cache,
                                   force_cache=force_cache)


@patch.object(redis, 'Redis', return_value=dict(abc=True))
class TestPatientDataHandler:
    @mongomock.patch(servers=('mongodb://localhost:27017'))
    def test_fetch_patient_specific_data(self, *args):
        insert_data_to_database()
        data_handler = get_patient_data_handler()
        features = data_handler.fetch_patient_specific_data(patient='TCGA-06-2557', modality='mRNA',
                                                            features=['TSPAN6'])
        assert features == [{'feature': 'TSPAN6', 'value': 14.05}]

        features = data_handler.fetch_patient_specific_data(patient='TCGA-06-2557', modality='mRNA',
                                                            features=['TSPAN6', 'TNMD'])
        assert features == [{'feature': 'TNMD', 'value': 0.219}, {'feature': 'TSPAN6', 'value': 14.05}]

    @mongomock.patch(servers=('mongodb://localhost:27017'))
    def test_set_patient_specific_data(self, *args):
        insert_data_to_database()
        data_handler = get_patient_data_handler(cache=True)
        with patch.object(PatientDataHandler, 'get_patient_info_from_mongo',
                          wraps=data_handler.get_patient_info_from_mongo) as mongo_spy:
            data_handler.fetch_patient_specific_data(patient='TCGA-06-2557',
                                                     modality='mRNA',
                                                     features=['TSPAN6'])
            mongo_spy.assert_called()
            mongo_spy.reset_mock()

            data_handler.fetch_patient_specific_data(patient='TCGA-06-2557',
                                                     modality='mRNA',
                                                     features=['TSPAN6'])
            mongo_spy.assert_not_called()
            mongo_spy.reset_mock()

            data_handler = get_patient_data_handler(cache=True, force_cache=True)
            data_handler.fetch_patient_specific_data(patient='TCGA-06-2557',
                                                     modality='mRNA',
                                                     features=['TSPAN6'])
            mongo_spy.assert_called_once()
            mongo_spy.reset_mock()

            data_handler.fetch_patient_specific_data(patient='TCGA-06-2557',
                                                     modality='mRNA',
                                                     features=['TNMD'])
            mongo_spy.assert_called_once()

    def test_generate_patient_modality_hash(self, *args):
        assert PatientDataHandler.generate_patient_modality_feature_hash(patient='TCGA-06-2557', modality='mRNA',
                                                                         feature='TNMD') == 'TCGA-06-2557-mRNA-TNMD'


@patch.object(redis, 'Redis', return_value=dict(abc=True))
class TestFeatureDataHandler:
    def test_generate_feature_modality_hash(self, *args):
        assert FeatureDataHandler.generate_feature_modality_hash(modality='mRNA', feature='TNMD') == 'mRNA-TNMD'

    @mongomock.patch(servers=('mongodb://localhost:27017'))
    def test_get_feature_specific_data(self, *args):
        insert_data_to_database()

        data_handler = get_feature_data_handler()

        data = data_handler.fetch_feature_specific_data(feature='TNMD', modality='mRNA')
        np.testing.assert_equal(data, np.array([0.219, 0.1109]))

    @mongomock.patch(servers=('mongodb://localhost:27017'))
    def test_set_feature_specific_data(self, *args):
        insert_data_to_database()
        data_handler = get_feature_data_handler(cache=True)
        with patch.object(FeatureDataHandler, 'get_feature_info_from_mongo',
                          wraps=data_handler.get_feature_info_from_mongo) as mongo_spy:
            data_handler.fetch_feature_specific_data(modality='mRNA',
                                                     feature='TNMD')
            mongo_spy.assert_called()
            mongo_spy.reset_mock()

            data_handler.fetch_feature_specific_data(modality='mRNA',
                                                     feature='TNMD')
            mongo_spy.assert_not_called()
            mongo_spy.reset_mock()

            data_handler = get_feature_data_handler(cache=True, force_cache=True)
            data_handler.fetch_feature_specific_data(modality='mRNA',
                                                     feature='TNMD')
            mongo_spy.assert_called_once()
            mongo_spy.reset_mock()

            data_handler.fetch_feature_specific_data(modality='mRNA',
                                                     feature='TSPAN6')
            mongo_spy.assert_called_once()


class TestFeaturesMetadataHandler:

    @pytest.mark.parametrize('patients,fetch_result', [(None, [{'feature': 'TNMD',
                                                                'metadata': {'max': 0.219, 'min': 0.1109, 'stdev': 2,
                                                                             'avg': 0.16494999999999999, 'var': 4.0}}]),
                                                       ('TCGA-06-2557', [{'feature': 'TNMD',
                                                                          'metadata': {'max': 0.219,
                                                                                       'min': 0.219,
                                                                                       'stdev': 1,
                                                                                       'avg': 0.219,
                                                                                       'var': 1.0}}])])
    def test_fetch_features_metadata(self, patients, fetch_result, *args):
        with mongomock.patch(servers=('mongodb://localhost:27017')):
            insert_data_to_database()
            with patch.object(redis, 'Redis', return_value=dict(abc=True)), patch.object(
                    FeaturesMetadataHandler, 'get_metadata_aggregation_values',
                    return_value={
                        '$group': {
                            '_id': '$name',
                            'max': {
                                '$max': '$value'
                            },
                            'min': {
                                '$min': '$value'
                            },
                            'stdev': {
                                '$sum': 1
                            },
                            'avg': {
                                '$avg': '$value'
                            }
                        }
                    }):
                data_handler = get_features_metadata_handler(cache=True)
                with patch.object(FeaturesMetadataHandler, 'get_features_metadata_from_mongo',
                                  wraps=data_handler.get_features_metadata_from_mongo) as mongo_spy:
                    results = data_handler.fetch_features_metadata(modality='mRNA',
                                                                   features=['TNMD'],
                                                                   patients=patients)
                    assert results == fetch_result
                    mongo_spy.assert_called()
                    mongo_spy.reset_mock()

                    results = data_handler.fetch_features_metadata(modality='mRNA',
                                                                   features=['TNMD'],
                                                                   patients=patients)
                    assert results == fetch_result
                    mongo_spy.assert_not_called()
                    mongo_spy.reset_mock()

                    data_handler = get_features_metadata_handler(cache=True, force_cache=True)
                    results = data_handler.fetch_features_metadata(modality='mRNA',
                                                                   features=['TNMD'],
                                                                   patients=patients)
                    assert results == fetch_result
                    mongo_spy.assert_called_once()
                    mongo_spy.reset_mock()

                    results = data_handler.fetch_features_metadata(modality='mRNA',
                                                                   features=['TSPAN6'],
                                                                   patients=patients)
                    mongo_spy.assert_called_once()

    def test_generate_feature_modality_metadata_hash(self):
        assert FeaturesMetadataHandler.generate_feature_modality_metadata_hash(modality='mRNA',
                                                                               feature='TNMD') == 'mRNA-TNMD-metadata'


@mongomock.patch(servers=('mongodb://localhost:27017'))
@patch.object(redis, 'Redis', return_value=dict(abc=True))
def test_generate_metadata(*args):
    with TemporaryDirectory() as t, patch.object(
            FeaturesMetadataHandler, 'get_metadata_aggregation_values',
            return_value={
                '$group': {
                    '_id': '$name',
                    'max': {
                        '$max': '$value'
                    },
                    'min': {
                        '$min': '$value'
                    },
                    'stdev': {
                        '$sum': 1
                    },
                    'avg': {
                        '$avg': '$value'
                    }
                }
            }):
        insert_data_to_database()

        with Path(t, 'patients.csv').open('w') as f:
            csv.writer(f).writerow(['TCGA-06-2557', 'TCGA-06-2559'])

        with Path(t, 'features.csv').open('w') as f:
            f.write('TSPAN6')

        metadata = generate_metadata(mongodb_connection_string='mongodb://localhost:27017',
                                     redis_host='redis://localhost:6379',
                                     modality='mRNA',
                                     patients_file=Path(t, 'patients.csv'),
                                     features_file=Path(t, 'features.csv'),
                                     db_name='TCGAOmics'
                                     )

        assert metadata == {'feature': 'TSPAN6',
                            'metadata': {'max': 16.7648, 'min': 14.05, 'stdev': 2, 'avg': 15.4074, 'var': 4.0}}
