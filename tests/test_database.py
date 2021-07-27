# Created by ohad at 7/27/21
import os.path

import pytest
from loguru import logger

from src.database import get_segmentation_files, get_dcm_dirs, get_unique_patient_barcodes, get_series_uids
from common.config import get_config


@pytest.fixture
def db_config():
    return get_config(os.path.join(os.path.dirname(__file__), '../config.toml'), 'brca-database')


def test_get_segmentation_files(db_config):
    segmentation_files = get_segmentation_files('TCGA-AO-A12D', db_config=db_config)
    assert isinstance(segmentation_files, list)


def test_get_dcm_dirs(db_config):
    dcm_dirs = get_dcm_dirs('TCGA-AO-A12D', db_config=db_config)
    assert isinstance(dcm_dirs, list)


def test_get_unique_patient_barcodes(db_config):
    assert isinstance(get_unique_patient_barcodes(collection_name='segmentation_files', db_config=db_config), list)


def test_get_series_uids(db_config):
    assert isinstance(get_series_uids(collection_name='tcga_breast_radiologist_reads',
                                      patient_barcode='TCGA-AO-A12D',
                                      db_config=db_config),
                      list)
