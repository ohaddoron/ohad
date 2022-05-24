import mongomock
import pymongo
import pytest
from pathlib import Path
import redis
from bson import json_util

from src.data_handler import DataHandler
from unittest.mock import patch, MagicMock


def insert_data_to_database():
    items = json_util.loads(Path(__file__).parent.joinpath('../resources/test_data_handler/mRNA.json').open().read())
    with pymongo.MongoClient('mongodb://localhost:27017') as client:
        client['TCGAOmics']['mRNA'].insert_many(items)


def get_data_handler(cache: bool = False, force_cache: bool = False) -> DataHandler:
    return DataHandler(mongodb_connection_string='mongodb://localhost:27017', db_name='TCGAOmics',
                       redis_connection_string='redis://localhost:6379', cache=cache, force_cache=force_cache)


@patch.object(redis, 'Redis', return_value=dict(abc=True))
class TestDataHandler:
    @mongomock.patch(servers=('mongodb://localhost:27017'))
    def test_fetch_patient_specific_data(self, *args):
        insert_data_to_database()
        data_handler = get_data_handler()
        features = data_handler.fetch_patient_specific_data(patient='TCGA-06-2557', modality='mRNA',
                                                            features=['TSPAN6'])
        assert features == [{'feature': 'TSPAN6', 'value': 14.05}]

        features = data_handler.fetch_patient_specific_data(patient='TCGA-06-2557', modality='mRNA',
                                                            features=['TSPAN6', 'TNMD'])
        assert features == [{'feature': 'TNMD', 'value': 0.219}, {'feature': 'TSPAN6', 'value': 14.05}]

    @mongomock.patch(servers=('mongodb://localhost:27017'))
    def test_set_patient_specific_data(self, *args):
        insert_data_to_database()
        data_handler = get_data_handler(cache=True)
        with patch.object(DataHandler, 'get_patient_info_from_mongo',
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

            data_handler = get_data_handler(cache=True, force_cache=True)
            data_handler.fetch_patient_specific_data(patient='TCGA-06-2557',
                                                     modality='mRNA',
                                                     features=['TSPAN6'])
            mongo_spy.assert_called_once()
            mongo_spy.reset_mock()

            data_handler.fetch_patient_specific_data(patient='TCGA-06-2557',
                                                     modality='mRNA',
                                                     features=['TNMD'])
            mongo_spy.assert_called_once()

    def test_generate_hash(self, *args):
        assert DataHandler.generate_patient_modality_feature_hash(patient='TCGA-06-2557', modality='mRNA',
                                                                  feature='TNMD') == 'TCGA-06-2557-mRNA-TNMD'
