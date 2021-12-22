import json
import os
from concurrent.futures import ProcessPoolExecutor

from common.database import init_database
from common.analytics import analyze_bundle
from scipy.stats import ks_2samp

from common.client import *

db = init_database('ohad')
results = analyze_bundle(mutation='BRCA1', mutation_status=True, col='CopyNumber', age_cutoff=45)
db['CopyNumberAnalysis'].insert_many(results)
results = analyze_bundle(mutation='BRCA1', mutation_status=False, col='CopyNumber', age_cutoff=45)
db['CopyNumberAnalysis'].insert_many(results)
