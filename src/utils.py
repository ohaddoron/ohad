from common import config
import os
from loguru import logger
from common.utils import init_cached_database, parse_mongodb_connection_string

import typing as tp


def get_config(name: tp.Optional[str] = None):
    config_path = os.path.join(os.path.dirname(__file__), '../config.toml')
    logger.debug(f'Reading {name} from {config_path}')
    return config.get_config(config_path, name)


def connect_to_database(config_name: tp.Optional[str] = None):
    config_name = config_name or 'brca-database'
    conf = get_config(config_name)
    return init_cached_database(parse_mongodb_connection_string(
        **conf), db_name=conf["db_name"])


def get_segmentation_files(bcr_patient_barcode: str, config_name: tp.Optional[str] = None):
    config_name = config_name or 'brca-database'
    db = connect_to_database(config_name=config_name)

    return [item['segmentation_file'] for item in db['segmentation_files'].find({'bcr_patient_barcode': bcr_patient_barcode})]


def get_dcm_dirs(bcr_patient_barcode: str, config_name: tp.Optional[str]=None):
    config_name = config_name or 'brca-database'
    db = connect_to_database(config_name=config_name)

    return [item['dcm_dir'] for item in (db['dcm_files'].find({'bcr_patient_barcode': bcr_patient_barcode}))]
