import numpy as np
from common.database import init_cached_database, parse_mongodb_connection_string
from src.utils import get_config, parse_dataframe_to_database
import pandas as pd
from tqdm import tqdm
from dask import dataframe as dd
from dask.delayed import delayed
from loguru import logger
import typer




def main(col_name: str, file_path: str):
    config = get_config('omics-database')
    db = init_cached_database(parse_mongodb_connection_string(**config), db_name=config['db_name'])
    # col_name = "ExonExpression"
    db[col_name].drop()
    db[col_name].create_index([('patient',  1)])

    df = dd.read_csv(file_path, sep='\t')
    patients = df.columns[1:]
    #%%
    print(df.head())
    #%%
    parse_dataframe_to_database(col=db[col_name], df=df, patients=patients)



if __name__ == '__main__':
    typer.run(main)