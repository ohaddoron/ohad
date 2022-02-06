from pathlib import Path

import pytest
from src.multi_omics import MultiOmicsRegressor, MultiOmicsRegressorConfig, DataConfig
import pickle as pkl


@pytest.fixture
def batch():
    return pkl.load(Path(Path(__file__).parent, '../resources/multi_omics_batch.pkl').open('rb'))


@pytest.fixture
def model():
    data_config = DataConfig()
    return MultiOmicsRegressor(modalities_model_config=MultiOmicsRegressorConfig().modalities_model_def,
                               train_patients=data_config.train_patients,
                               val_patients=data_config.val_patients,
                               **MultiOmicsRegressorConfig().dict())


def test_train_step(model: MultiOmicsRegressor, batch):
    model.training_step(batch=batch, batch_idx=0)
