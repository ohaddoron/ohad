# %%
import streamlit as st
from common.database import init_database
import httpx
import numpy as np
import pandas as pd
from plotly import express as px

# %%


AGE_CUTOFF = 45
# %%
db = init_database('omics-database')


# %%
def get_patients_with_mutation(mutation):
    with httpx.Client() as client:
        r = client.get('http://medical001-5.tau.ac.il/api/patients_by_mutation', params=dict(mutation=mutation))
        r.raise_for_status()
        patients = r.json()

        r = client.post('http://medical001-5.tau.ac.il/api/patients_age', json=patients)
        r.raise_for_status()
        patients_age = r.json()
    return patients_age


def get_patients_without_mutation(mutation):
    with httpx.Client() as client:
        r = client.get('http://medical001-5.tau.ac.il/api/patients_by_mutation',
                       params=dict(mutation=mutation, mutation_status=False))
        r.raise_for_status()
        patients = r.json()

        r = client.post('http://medical001-5.tau.ac.il/api/patients_age', json=patients)
        patients_age = r.json()
    return patients_age


def split_by_age_cutoff(patients, cutoff):
    early = [patient for patient in patients if patient['age'] <= cutoff]
    late = [patient for patient in patients if patient['age'] > cutoff]
    return early, late


def extract_age(items):
    return [item['age'] for item in items]


# %%
for mutation in ['BRCA1', 'TP53']:
    st.title(mutation)
    mutation_patients = get_patients_with_mutation(mutation=mutation)
    non_mutation_patients = get_patients_without_mutation(mutation=mutation)

    # %%

    early_mutation, late_mutation = split_by_age_cutoff(mutation_patients, AGE_CUTOFF)
    early_non, late_non = split_by_age_cutoff(non_mutation_patients, AGE_CUTOFF)
    # %%
    early_mutation = extract_age(early_mutation)
    late_mutation = extract_age(late_mutation)
    early_non = extract_age(early_non)
    late_non = extract_age(late_non)
    # %%

    df = pd.DataFrame(dict(
        series=np.concatenate((["early_mutation"] * len(early_mutation), ["late_mutation"] * len(late_mutation),
                               ['early_non'] * len(early_non), ['late_non'] * len(late_non))),
        data=np.concatenate((early_mutation, late_mutation, early_non, late_non))
    ))
    # %%
    st.plotly_chart(px.histogram(df, x="data", color="series", barmode="overlay"))

# %%


# %%
