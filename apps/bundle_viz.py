import pandas as pd
import plotly.express as px

from common.database import init_database
from tqdm import tqdm
import streamlit as st


# %%


# %%

@st.cache(persist=True)
def get_samples_from_column(col: str):
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

    return d


def main():
    st.title('Bundle Plot')
    viz_items = pd.DataFrame(gather_data()).T
    viz_items = viz_items.rename(columns={'_id': 'patient'})

    # viz_items[viz_items > 5] = 5
    arr = viz_items.to_xarray().transpose()
    fig = px.imshow(arr.to_array().transpose(), width=900, height=3000, zmax=3, contrast_rescaling='infer')

    st.plotly_chart(fig)


if __name__ == '__main__':
    main()
