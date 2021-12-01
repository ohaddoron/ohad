import pandas as pd
import plotly.express as px

from common.database import init_database
from tqdm import tqdm
import streamlit as st
from sklearn.decomposition import PCA
import xarray


# %%


# %%

@st.cache(persist=True)
def get_samples_from_column(col: str) -> pd.DataFrame:
    db = init_database('omics-database')
    samples = db[col].distinct('sample')
    return samples


def gather_data():
    db = init_database('omics-database')
    cols = db.list_collection_names()
    #
    cols = [col for col in cols if col not in ['CopyNumberSegmentsAfterRemove'
                                               'GenomicSegment',
                                               'GenomicSegmentBeforeRemoval',
                                               'CopyNumberSegmentsBeforeRemove',
                                               'SomaticMutationPV',
                                               'ClinicalBRCA1']
            ]
    patients = set()
    for col in cols:
        cur_patients = db[col].distinct('patient')
        patients.update(set(cur_patients))
    patients = list(patients)
    d = {patient: {col: 0 for col in cols} for patient in patients}
    for col in tqdm(cols):
        samples = get_samples_from_column(col)

        patients = list([sample[:12] for sample in samples])

        for patient in list(set(patients)):
            d[patient][col] = patients.count(patient)
    df = pd.DataFrame(d).T
    df = df.reindex((df > 0).mean().sort_values(ascending=False).index, axis=1)
    return df


def gather_data_pca() -> pd.DataFrame:
    df = gather_data()
    df_normalized = (df - df.mean()) / df.std()

    pca = PCA(n_components=df.shape[1])
    pca.fit(df_normalized)

    loadings = pd.DataFrame(pca.components_.T,
                            columns=['PC%s' % _ for _ in range(1, len(df_normalized.columns) + 1)],
                            index=df.columns)

    return xarray.DataArray(data=pca.transform(df_normalized),
                            dims=['Patient', 'PC'],
                            coords=dict(
                                PC=['PC%s' % _ for _ in range(1, len(df_normalized.columns) + 1)],
                                Patient=df.index.tolist()
                            )).to_pandas()


def display_dataframe(df, zmax=3, zmin=0):
    arr = df.to_xarray().transpose()
    fig = px.imshow(arr.to_array().transpose(), width=900, height=3000, zmax=zmax, zmin=zmin,
                    contrast_rescaling='infer')
    st.plotly_chart(fig)


def main():
    st.title('Bundle Plot')
    df = gather_data()
    display_dataframe(df)
    arr_pca = gather_data_pca()
    display_dataframe(arr_pca, zmax=6, zmin=-6)
    pass


if __name__ == '__main__':
    main()
