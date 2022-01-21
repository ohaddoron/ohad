import pytest
from torch.utils.data import DataLoader

from src.dataset import AttributeFillerDataset, MultiOmicsDataset


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

    assert {'attributes', 'dropped_attributes_index', 'dropped_attributes', 'targets'} == set(items.keys())


class TestMultiOmicsDataset:
    @pytest.fixture
    def ds(self, patients):
        return MultiOmicsDataset(patients=patients, collections=['GeneExpression', 'CopyNumber'])

    def test_get_standardization_dict_multiple_collections(self, ds: MultiOmicsDataset):
        assert isinstance(ds.standardization_dicts, dict)
        assert {'GeneExpression', 'CopyNumber'} - set(ds.standardization_dicts.keys()) == set()

    def test_define_samples(self, ds: MultiOmicsDataset):
        assert all([len(sample) == 3 for sample in ds.samples])
        assert len(ds.samples) == 12
