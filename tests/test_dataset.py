import pytest
from torch.utils.data import DataLoader

from src.dataset import AttributeFillerDataset


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


@pytest.mark.parametrize('standardize', [True, False])
def test_attribute_filler_dataset(patients, standardize):
    ds = AttributeFillerDataset(patients=patients, collection_name='GeneExpression', attributes_drop_rate=0.2,
                                standardize=standardize)

    item = ds[0]

    dl = DataLoader(dataset=ds, batch_size=4, num_workers=1)

    items = next(iter(dl))

    assert {'attributes', 'dropped_attributes_index', 'dropped_attributes', 'targets'} == set(items.keys())
