from loguru import logger

from common.database import init_database, fetch_collection_as_table
import pandas as pd
from umap import UMAP
from matplotlib import pyplot as plt
from common.analytics import get_patients_split
import numpy as np
import streamlit as st
import typing as tp


def get_patients_by_mutation(mutation: str):
    early_pos, late_pos = get_patients_split(mutation, mutation_status=True)
    early_neg, late_neg = get_patients_split(mutation, mutation_status=False)
    patients = early_pos + late_pos + early_neg + late_neg

    return patients, early_pos, late_pos, early_neg, late_neg


@st.cache
def _get_dataframe(col: str):
    logger.info(f'Fetching {col}')
    df = pd.DataFrame(fetch_collection_as_table(col=col, config_name='omics-database'))
    logger.info(f'{col} fetched')
    return df


def get_dataframe(patients: tp.List, col: str):
    df = _get_dataframe(col)
    df = df[df.patient.isin(patients)]
    df = df.set_index('patient').drop('sample', axis=1)
    return df


def fit_and_index(df: pd.DataFrame, early_pos, late_pos, early_neg, late_neg):
    reducer = UMAP(random_state=0)
    reducer.fit(np.array(df))
    embedding = reducer.transform(np.array(df))
    indices = []
    for patient in df.index:
        indices.append([patient in lst for lst in (early_pos, late_pos, early_neg, late_neg)].index(True))

    return embedding, indices


def create_figure(embedding, indices, col):
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=indices, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(5) - 0.5).set_ticks(np.arange(4))
    plt.title(f'UMAP projection of the {col}', fontsize=24)
    return plt.gcf()


def main():
    cols = ['GeneExpression', 'CopyNumber', 'PathwayActivity']

    with st.sidebar.container():
        mutation = st.selectbox(label='mutation', options=['BRCA1', 'TP53'])
    for col in cols:
        patients, early_pos, late_pos, early_neg, late_neg = get_patients_by_mutation(mutation=mutation)
        df = get_dataframe(patients, col=col)
        embeddings, indices = fit_and_index(df, early_pos, late_pos, early_neg, late_neg)
        fig = create_figure(embeddings, indices, col)

        st.pyplot(fig)


main()
