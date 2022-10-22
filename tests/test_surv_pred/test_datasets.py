import numpy as np
import pytest
from torch.utils.data import DataLoader

from surv_pred.datasets import ModalitiesDataset


@pytest.fixture(scope='class')
def config():
    return {
        'patients': ['TCGA-04-1331', 'TCGA-04-1332', 'TCGA-04-1336',
                     'TCGA-04-1337', 'TCGA-04-1341', 'TCGA-04-1342',
                     'TCGA-04-1343', 'TCGA-04-1346', 'TCGA-04-1347',
                     'TCGA-04-1348', 'TCGA-04-1349', 'TCGA-04-1350',
                     ],
        'modality': 'Clinical',
        'db_params': {
            'mongodb_connection_string': 'mongodb://admin:mimp1lab@132.66.207.18:80/?authSource=admin&authMechanism=SCRAM-SHA-256&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false',
            'db_name': 'TCGAOmics'
        }

    }


@pytest.fixture(scope='class')
def dataset(config):
    return ModalitiesDataset(**config)


class TestModalitiesDataset:
    def test_feature_names(self, dataset: ModalitiesDataset):
        feature_names = dataset.feature_names
        assert len(feature_names) == 1876

    def test_fetch_item(self, dataset: ModalitiesDataset):
        item = dataset[0]
        assert isinstance(item, dict)
        assert isinstance(item['features'], np.ndarray)

    def test_fetch_batch(self, dataset: ModalitiesDataset):
        dl = DataLoader(dataset, batch_size=4)
        batch = next(iter(dl))
        assert len(batch['features']) == 4
