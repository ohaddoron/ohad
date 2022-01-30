import pytest
from torch.utils.data import DataLoader

from src.dataset import AttributeFillerDataset, MultiOmicsDataset, AttentionMixin


@pytest.fixture
def patients():
    return ['TCGA-3C-AAAU',
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
            'TCGA-A1-A0SF']


def test_attribute_filler_dataset(patients):
    ds = AttributeFillerDataset(patients=patients, collection_name='GeneExpression', attributes_drop_rate=0.2)

    item = ds[0]

    dl = DataLoader(dataset=ds, batch_size=4, num_workers=1)

    items = next(iter(dl))

    assert {'attributes', 'dropped_attributes_index', 'dropped_attributes', 'targets', 'attributes_names'} == set(
        items.keys())


class TestMultiOmicsDataset:
    @pytest.fixture
    def ds(self, patients):
        return MultiOmicsDataset(patients=patients, collections=['GeneExpression', 'CopyNumber'])

    def test_define_samples(self, ds: MultiOmicsDataset):
        assert all([len(sample) == 3 for sample in ds.samples])
        assert len(ds.samples) == 12
        assert all([isinstance(sample, tuple) for sample in ds.samples])
        assert all([all([isinstance(item, dict) for item in sample]) for sample in ds.samples])

    def test_get_samples(self, ds):
        item = ds[10]
        assert isinstance(item, dict)
        assert set(item.keys()) == {'anchor', 'pos', 'neg'}

        dl = DataLoader(dataset=ds, batch_size=12, num_workers=0)
        items = next(iter(dl))
        pass


class TestAttentionMixin:
    def test_get_sample(self, patients):
        import torch
        class AttributeFillerAttentionDataset(AttentionMixin, AttributeFillerDataset): pass

        ds = AttributeFillerAttentionDataset(patients=patients,
                                             collection_name='GeneExpression',
                                             attributes_drop_rate=0.2)

        dl = DataLoader(dataset=ds, batch_size=2, num_workers=0)

        item = next(iter(dl))
        assert item['attributes'].shape == torch.Size((2, 2, 17814))
