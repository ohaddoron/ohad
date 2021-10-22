import numpy as np
from common.database import init_cached_database, parse_mongodb_connection_string
from src.utils import get_config, parse_dataframe_to_database
import pandas as pd
from tqdm import tqdm
from dask import dataframe as dd
from dask.delayed import delayed
from loguru import logger
import typer




def main(col_name: str, file_path: str, num_rows_to_parse_before_dump: int = 100000):
    config = get_config('omics-database')
    db = init_cached_database(parse_mongodb_connection_string(**config), db_name=config['db_name'])
    # col_name = "ExonExpression"
    db[col_name].drop()
    db[col_name].create_index([('patient',  1)])
    db[col_name].create_index([('name',  1)])
    db[col_name].create_index([('patient',  1), ('name', 1)], unique=True)

    df = dd.read_csv(file_path, sep='\t', sample=2560000)
    patients = df.columns[1:]
    #%%
    print(df.head())
    #%%
    parse_dataframe_to_database(col=db[col_name], df=df, patients=patients, num_rows_to_parse_before_dump=num_rows_to_parse_before_dump)



if __name__ == '__main__':
    typer.run(main)