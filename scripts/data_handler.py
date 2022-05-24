import asyncio

import httpx
import typing as tp

import numpy as np
from loguru import logger
from tqdm import tqdm

API_PATH = 'http://medical001-5.tau.ac.il/dev/api'


def get_patients_split(mutation, mutation_status):
    with httpx.Client() as client:
        r = client.get(f'{API_PATH}/patients_by_mutation',
                       params=dict(mutation=mutation, mutation_status=mutation_status))
        r.raise_for_status()

        patients = r.json()
        r = client.post(f'{API_PATH}/patients_age', json=patients)

        r.raise_for_status()
        patients_ages = r.json()

    early = [item['patient'] for item in patients_ages if item['age'] < 45]
    late = [item['patient'] for item in patients_ages if item['age'] >= 45]
    return early, late


def get_feature_names(col):
    with httpx.Client() as client:
        r = client.get(f'{API_PATH}/feature_names',
                       params=dict(col=col))
        r.raise_for_status()
    return r.json()


async def get_patients_data_for_feature(collection: str, patients: tp.List[str], feature_name: str, client):
    r = await client.post(f'{API_PATH}/features_for_patients',
                          json=dict(col=collection, patients=patients, feature_name=feature_name))
    try:
        r.raise_for_status()
    except:
        return None
    return r.json()


async def get_patients_data_all_features(collection: str, patients: tp.List[str]):
    feature_names = get_feature_names(col=collection)
    results = []
    chunks = np.arange(0, len(feature_names), 1000)

    async with httpx.AsyncClient(timeout=None) as client:
        for i in range(len(chunks) - 1):
            for future in tqdm(asyncio.as_completed(
                    map(lambda feature_name: get_patients_data_for_feature(collection=collection, patients=patients,
                                                                           feature_name=feature_name, client=client),
                        feature_names[chunks[i]:chunks[i + 1]]))):
                result = await future
                results += result

    return result
