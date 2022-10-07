from pathlib import Path
from unittest.mock import patch

import json
import mongomock
import numpy as np
import pymongo
import pytest
import redis
import torch
from bson import json_util
from fakeredis import FakeRedis, FakeStrictRedis
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from src.dataset import AttributeFillerDataset, MultiOmicsDataset, AttentionMixin


@pytest.fixture
def patients():
    return list({
        'TCGA-3C-AAAU',
        'TCGA-3C-AALI',
        'TCGA-3C-AALJ',
        'TCGA-3C-AALK',
        'TCGA-4H-AAAK',
        'TCGA-5L-AAT0',
        'TCGA-5L-AAT1',
        'TCGA-5T-A9QA',
        'TCGA-A1-A0SB',
        'TCGA-A1-A0SD',
        'TCGA-A1-A0SE',
        'TCGA-A1-A0SF',
        'TCGA-3C-AAAU',
        'TCGA-3C-AALI',
        'TCGA-3C-AALJ',
        'TCGA-3C-AALK',
        'TCGA-4H-AAAK',
        'TCGA-5L-AAT0',
        'TCGA-5L-AAT1',
        'TCGA-5T-A9QA',
        'TCGA-A1-A0SB',
        'TCGA-A1-A0SD',
        'TCGA-A1-A0SE',
        'TCGA-A1-A0SF',
        'TCGA-A1-A0SG',
        'TCGA-A1-A0SH',
        'TCGA-A1-A0SI',
        'TCGA-A1-A0SJ',
        'TCGA-A1-A0SK',
        'TCGA-A1-A0SM',
        'TCGA-A1-A0SN',
        'TCGA-A1-A0SO',
        'TCGA-A1-A0SP',
        'TCGA-A1-A0SQ',
        'TCGA-A2-A04N',
        'TCGA-A2-A04P',
        'TCGA-A2-A04Q'
    })


def test_attribute_filler_dataset(patients):
    ds = AttributeFillerDataset(patients=patients, collection_name='GeneExpression', attributes_drop_rate=0.2,
                                config_name='brca-reader', override_raw_attributes_file=True)

    item = ds[0]

    dl = DataLoader(dataset=ds, batch_size=4, num_workers=1)

    items = next(iter(dl))

    assert {'attributes', 'dropped_attributes_index', 'dropped_attributes', 'targets', 'attributes_names'} == set(
        items.keys())


class TestMultiOmicsDataset:
    @pytest.fixture
    def ds(self, patients):
        return MultiOmicsDataset(patients=patients, collections=['GeneExpression', 'CopyNumber'],
                                 config_name='brca-reader', get_from_redis=True)

    def test_get_samples(self, ds):
        dl = DataLoader(dataset=ds,
                        num_workers=0,
                        batch_size=None,
                        sampler=BatchSampler(sampler=RandomSampler(data_source=ds), batch_size=4, drop_last=True),
                        collate_fn=None
                        )
        batch = next(iter(dl))
        assert isinstance(batch, dict)
        assert set(batch.keys()) == {'pos_survival', 'anchor_modality', 'anchor', 'anchor_survival', 'neg_modality',
                                     'neg', 'pos', 'pos_modality', 'neg_survival'}
        for value in batch.values():
            assert isinstance(value, (torch.Tensor, str))


class TestAttentionMixin:
    def test_get_sample(self, patients):
        import torch
        class AttributeFillerAttentionDataset(AttentionMixin, AttributeFillerDataset): pass

        ds = AttributeFillerAttentionDataset(patients=patients,
                                             collection_name='GeneExpression',
                                             attributes_drop_rate=0.2,
                                             config_name='brca-reader')

        dl = DataLoader(dataset=ds, batch_size=2, num_workers=0)

        item = next(iter(dl))
        assert item['attributes'].shape == torch.Size((2, 2, 19672))


@pytest.fixture(scope='class')
def setup_database():
    with mongomock.patch(servers='localhost'):
        with pymongo.MongoClient() as client:
            for file in [Path(__file__).parent.joinpath('../resources/miRNA.json'),
                         Path(__file__).parent.joinpath('../resources/mRNA.json'),
                         Path(__file__).parent.joinpath('../resources/DNAm.json'),
                         Path(__file__).parent.joinpath('../resources/metadata.json'),
                         Path(__file__).parent.joinpath('../resources/survival.json')]:
                with file.open() as f:
                    metadata = json_util.loads(f.read())
                with pymongo.MongoClient() as client:
                    client['TCGA'][file.stem].insert_many(metadata)

        yield


@patch.object(redis, 'StrictRedis', side_effect=FakeStrictRedis)
class TestAttributeDataset:
    def test_get_train_val_split(self, redis, setup_database, *args):
        from src.dataset import AttributesDataset

        out = AttributesDataset.get_train_val_split(mongodb_connection_string='mongodb://localhost:27017',
                                                    db_name='TCGA',
                                                    metadata_collection_name='metadata')
        train_patients, val_patients = out.values()

        assert not set(train_patients).intersection(val_patients)

    def test_getitem(self, redis, setup_database, *args):
        from src.dataset import AttributesDataset

        ds = AttributesDataset(mongodb_connection_string='mongodb://localhost:27017',
                               db_name='TCGA',
                               modality='miRNA',
                               patients=None,
                               drop_rate=0.2)
        item = ds[0]
        assert isinstance(item, dict)
        assert np.mean((item['inputs'] - item['outputs']) ** 2) > 0
        assert 0. <= item['inputs'].max() <= 1.
        assert 0. <= item['outputs'].max() <= 1.

    def test_min_max_values(self, redis, setup_database, *args):
        from src.dataset import AttributesDataset
        value_per_feature = AttributesDataset.get_min_max_values(
            mongodb_connection_string='mongodb://localhost:27017',
            db_name='TCGA',
            modality='miRNA'
        )
        assert isinstance(value_per_feature, dict)

    def test_patients_are_according_to_intersect(self, redis, setup_database, *args):
        from src.dataset import AttributesDataset

        ds = AttributesDataset(mongodb_connection_string='mongodb://localhost:27017',
                               db_name='TCGA',
                               modality='miRNA',
                               patients=['TCGA-AR-A24N'],
                               drop_rate=0.2)

        assert ds._patients == ['TCGA-AR-A24N']

        ds = AttributesDataset(mongodb_connection_string='mongodb://localhost:27017',
                               db_name='TCGA',
                               modality='miRNA',
                               patients=None,
                               drop_rate=0.2)
        assert len(set(ds._patients) - {'TCGA-AR-A24N'}) > 0


@patch.object(redis, 'StrictRedis', side_effect=FakeStrictRedis)
class TestMultiOmicsAttributesDataset:
    def test_getitem(self, redis, setup_database, *args):
        from src.dataset import MultiOmicsAttributesDataset

        ds = MultiOmicsAttributesDataset(mongodb_connection_string='mongodb://localhost:27017',
                                         db_name='TCGA',
                                         modalities=['miRNA', 'DNAm', 'mRNA'],
                                         patients=None,
                                         features={'miRNA': None, 'DNAm': None,
                                                   'mRNA': json.load(
                                                       Path(__file__).parent.joinpath(
                                                           '../resources/mRNA-features.json').open())},
                                         drop_rate={'miRNA': 0.2, 'DNAm': 0.2, 'mRNA': 0.2}
                                         )
        item = ds[0]
        assert isinstance(item, list)
        assert len(item) == 3
        assert all([
            'modality' in item_.keys() and 'inputs' in item_.keys() and 'patient' in item_.keys() and 'outputs' in item_.keys()
            for item_ in item])

        dl = DataLoader(ds,
                        num_workers=0,
                        batch_size=2,
                        collate_fn=MultiOmicsAttributesDataset.batch_collate_fn(modalities=['miRNA', 'DNAm', 'mRNA']))
        item = next(iter(dl))
        assert isinstance(item, dict)
        assert {'miRNA', 'DNAm', 'mRNA'} == set(item.keys())

        for value in item.values():
            assert {'inputs', 'reconstruction_targets', 'idx', 'triplet_kind', 'project_id'} == set(value.keys())
