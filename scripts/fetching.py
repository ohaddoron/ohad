from ohad.src.data_fetching import get_patients_split
import pandas as pd

early_pos, late_pos = get_patients_split(mutation='BRCA1', mutation_status=True)
early_neg, late_neg = get_patients_split(mutation='BRCA1', mutation_status=False)

import asyncio
from src.data_fetching import get_patients_data_all_features

# %%

early_pos_res = asyncio.run(get_patients_data_all_features(collection='GeneExpression', patients=early_pos))
late_pos_res = asyncio.run(get_patients_data_all_features(collection='GeneExpression', patients=late_pos))
early_neg_res = asyncio.run(get_patients_data_all_features(collection='GeneExpression', patients=early_neg))
late_neg_res = asyncio.run(get_patients_data_all_features(collection='GeneExpression', patients=late_neg))

pass
