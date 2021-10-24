import numpy as np
from common.database import init_cached_database, parse_mongodb_connection_string
from src.utils import get_config, parse_file_to_database
import pandas as pd
from tqdm import tqdm
from dask import dataframe as dd
from dask.delayed import delayed
from loguru import logger
import typer

if __name__ == '__main__':
    typer.run(parse_file_to_database)
