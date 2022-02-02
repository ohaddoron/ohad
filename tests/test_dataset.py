import pytest
import torch
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
                                 config_name='brca-reader')

    def test_get_samples(self, ds):
        dl = DataLoader(dataset=ds,
                        num_workers=0,
                        batch_size=None,
                        sampler=BatchSampler(sampler=RandomSampler(data_source=ds), batch_size=4, drop_last=True),
                        collate_fn=None
                        )
        batch = next(iter(dl))
        assert isinstance(batch, dict)
        assert set(batch.keys()) == {'pos', 'anchor_modality', 'pos_modality', 'neg_modality', 'anchor', 'neg'}
        for value in batch.values():
            assert isinstance(value, torch.Tensor)


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
