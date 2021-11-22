from common.database import init_cached_database, init_database
from src.utils import parse_mongodb_connection_string, get_config
from tqdm import tqdm

# %%

config = get_config('omics-database')
db = init_cached_database(parse_mongodb_connection_string(
    **config), db_name=config['db_name'])

# %%
config = get_config('ohad')
ohad_db = init_cached_database(parse_mongodb_connection_string(
    **config), db_name=config['db_name'])
# %%
cols = db.list_collection_names()
#
cols = [col for col in cols if col not in ['CopyNumberSegmentsAfterRemove'
                                           'GenomicSegment',
                                           'GenomicSegmentBeforeRemoval',
                                           'CopyNumberSegmentsBeforeRemove']
        ]

patients = set()
ohad_db['patient_col_hist'].drop()
for col in cols:
    cur_patients = db[col].distinct('patient')
    if any([len(patient) > 12 for patient in cur_patients]):
        print(col)
    patients.update(set(cur_patients))
patients = list(patients)
# %%

d = {patient: {col: 0 for col in cols} for patient in patients}

# for patient in tqdm(patients):
for col in tqdm(cols):
    samples = db[col].distinct('sample')

    patients = list([sample[:12] for sample in samples])

    for patient in list(set(patients)):
        d[patient][col] = patients.count(patient)

# [d[patient][col] = patient
for key, value in d.items():
    ohad_db['patient_col_hist'].insert_one({'_id': key, **value})
