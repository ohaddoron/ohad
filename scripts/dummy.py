import math

import numpy as np
from common.database import init_cached_database, parse_mongodb_connection_string
from src.utils import get_config
import pandas as pd
from tqdm import tqdm
from dask import dataframe as dd
from dask.delayed import delayed

# %%

config = get_config('omics-database')
db = init_cached_database(parse_mongodb_connection_string(
    **config), db_name=config['db_name'])

# %%

col_name = "Phenotypes"
df = pd.read_csv(r"C:\Users\ogdor\Projects\data\RawData\20Phenotypes\Extracted\TCGA.BRCA.sampleMap_BRCA_clinicalMatrix",
                 sep='\t')

# %%

df.head()

# %%
import math

col = db[col_name]
col.drop()
col.create_index([('patient', 1)])
col.create_index([('name', 1)])
col.create_index([('patient', 1), ('name', 1)], unique=True)
aggregator = []

for i, row in df.iterrows():
    patient = row[0]
    out_dict = dict()
    for col_name in df.columns[1:]:
        # print(dict(col=val))
        val = df[col_name].iloc[i]
        if isinstance(val, float) and math.isnan(val):
            continue
        if isinstance(val, np.int64):
            val = int(val)
        out_dict = dict(patient=patient, name=col_name, value=val)

        aggregator.append(out_dict)
        if (len(aggregator) % 10000) == 0:
            col.insert_many(aggregator)
            # logger.info(f'Inserted {len(x.inserted_ids)} documents')
            aggregator = []
