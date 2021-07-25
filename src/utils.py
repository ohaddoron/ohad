from common import config
import os
from loguru import logger

import typing as tp


def get_config(name: tp.Optional[str] = None):
    config_path = os.path.join(os.path.dirname(__file__), '../config.toml')
    logger.debug(f'Reading {name} from {config_path}')
    return config.get_config(config_path, name)
