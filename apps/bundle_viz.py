import numpy as np
import pandas as pd
import plotly.express as px

from common.database import init_database
from tqdm import tqdm
import streamlit as st
from sklearn.decomposition import PCA
import xarray


# %%


# %%

@st.cache
def get_samples_from_column(col: str) -> pd.DataFrame:
    db = init_database('omics-database')
    samples = db[col].distinct('sample')
    return samples


@st.cache
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
    fig = px.imshow(arr.to_array().transpose(), width=900, height=700, zmax=zmax, zmin=zmin,
                    contrast_rescaling='infer')
    st.plotly_chart(fig)


def type_count(df):
    cols = st.columns(3)
    nbundles = int(cols[0].number_input('Number of Bundles', min_value=10, max_value=df.shape[1], value=df.shape[1]))
    log_log = cols[1].checkbox(label='Log-Log plot', value=True)
    # random_selection = cols[2].checkbox(label='Randomly Select Bundles', value=False)
    #
    # if random_selection:
    df = df.iloc[:, :nbundles]
    urows = np.unique(df.to_numpy(), axis=0)
    sums = []
    for urow in urows:
        sums.append(sum([all(item) for item in (df.to_numpy() == urow)]))
    sums = sorted(sums, reverse=True)
    df_ = pd.DataFrame(dict(Index=list(np.arange(0, len(sums))), Count=sums))
    fig = px.line(df_, x="Index", y="Count", log_y=log_log, log_x=log_log)
    st.plotly_chart(fig)

    cumsum = np.cumsum(sums)

    df_ = pd.DataFrame(dict(BundleIndex=list(np.arange(0, len(sums))), CumulativeSum=cumsum / cumsum[-1]))
    fig = px.line(df_, x='BundleIndex', y='CumulativeSum')
    st.plotly_chart(fig)


@st.cache
def get_patients_from_collection(collection):
    db = init_database('omics-database')
    return db[collection].distinct('patient')


def patient_count_according_to_columns(df):
    counts = []
    patients = set()
    for col in df.columns[::-1]:
        patients.update(get_patients_from_collection(collection=col))
        counts.append(len(patients))

    counts = [100 * count / max(counts) for count in counts]

    df_ = pd.DataFrame({'#Bundles': list(np.arange(1, len(counts) + 1)), '% Patients': counts})
    fig = px.line(df_, x='#Bundles', y='% Patients')
    st.plotly_chart(fig)


def patient_representation_per_bundle(df, num_patients):
    out = dict(Bundle=[], Percentage=[])
    for col in df.columns[::-1]:
        out['Bundle'].append(col)
        out['Percentage'].append(100 * len(get_patients_from_collection(collection=col)) / num_patients)

    df_ = pd.DataFrame({'Bundle': out['Bundle'], '% Patients': out['Percentage']})
    fig = px.bar(df_, x='Bundle', y='% Patients')
    st.plotly_chart(fig)


def main():
    st.header('Bundle Plot')
    df = gather_data()
    display_dataframe(df)
    st.header('Occurrence Analysis')
    type_count(df)
    st.header('Patients Count')
    patient_count_according_to_columns(df)
    patient_representation_per_bundle(df, num_patients=1101)

    pass


if __name__ == '__main__':
    main()
