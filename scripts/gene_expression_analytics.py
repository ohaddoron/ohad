import json
import os
from concurrent.futures import ProcessPoolExecutor

from common.database import init_database
from scipy.stats import ks_2samp

from common.client import *


def get_patients_split(mutation, mutation_status):
    patients = get_patients_by_mutation(mutation=mutation, mutation_status=mutation_status)
    patients_ages = get_patients_age(patients=patients)

    early = [item['patient'] for item in patients_ages if item['age'] < 45]
    late = [item['patient'] for item in patients_ages if item['age'] >= 45]
    return early, late


def test_significance(patients_1: tp.List[str], patients_2: tp.List[str], gene: str):
    data_1 = [item['value'] for item in
              get_features_for_patients(col='GeneExpression', feature_name=gene, patients=patients_1)]
    data_2 = [item['value'] for item in
              get_features_for_patients(col='GeneExpression', feature_name=gene, patients=patients_2)]
    return dict(gene=gene, pvalue=float(ks_2samp(data1=data_1, data2=data_2).pvalue))


genes = get_feature_names('GeneExpression')

early_pos, late_pos = get_patients_split(mutation='TP53', mutation_status=True)
early_neg, late_neg = get_patients_split(mutation='TP53', mutation_status=False)


def test_positive(gene):
    out = test_significance(patients_1=early_pos, patients_2=late_pos, gene=gene)
    out['TP53'] = True
    return out


def test_negative(gene):
    out = test_significance(patients_1=early_neg, patients_2=late_neg, gene=gene)
    out['TP53'] = False
    return out


db = init_database(config_name='ohad')

with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    results = list(executor.map(test_positive, genes))
    db['GeneExpressionAnalysis'].insert_many(results)
    results = list(executor.map(test_negative, genes))
    db['GeneExpressionAnalysis'].insert_many(results)
