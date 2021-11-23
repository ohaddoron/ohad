import asyncio
import pickle
import pprint
import tempfile
from functools import lru_cache
from http.client import HTTPException
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import aiofiles
import httpx
import skimage
from PIL import Image
from bson import json_util
from fastapi import FastAPI, Query
from motor import MotorDatabase
from pydantic import BaseModel
import typing as tp
from fastapi_pagination import Page, add_pagination, paginate
from starlette.background import BackgroundTask, BackgroundTasks
from starlette.requests import Request
from starlette.responses import RedirectResponse, JSONResponse, StreamingResponse, Response
from aiocache import cached
from common.database import parse_mongodb_connection_string
from common.utils import read_dicom_images
from src.utils import init_cached_database, get_config
import orjson
import skimage.exposure


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: tp.Any) -> bytes:
        return orjson.dumps(content)


class Document(BaseModel):
    patient: str
    data: dict


class BSONResponse(JSONResponse):
    media_type = 'application/json'

    def render(self, content: tp.Any) -> bytes:
        return json_util.dumps(content)


# app = FastAPI(title='Fetcher', default_response_class=BSONResponse)
app = FastAPI(title='Fetcher', default_response_class=ORJSONResponse)


@lru_cache
def init_database(async_flag=True) -> MotorDatabase:
    config = get_config('omics-database')
    db = init_cached_database(parse_mongodb_connection_string(
        **config), db_name=config['db_name'], async_flag=async_flag)
    return db


@cached(ttl=600)
async def query_db(patients: tp.List[str]):
    db = init_database()
    data = []
    cursor = db['ClinicalData'].aggregate(
        [{
             "$match": {
                 'case_submitter_id':
                     {"$in":
                          patients
                      }
             }
         } if patients else {"$match":
                                 {}
                             },
         {
             '$addFields': {
                 'patient': '$case_submitter_id'
             }
         }, {
             '$lookup': {
                 'from': 'ClinicalBRCA1',
                 'localField': 'case_submitter_id',
                 'foreignField': 'case_submitter_id',
                 'as': 'BRCA1'
             }
         }, {
             '$lookup': {
                 'from': 'GeneExpression',
                 'localField': 'case_submitter_id',
                 'foreignField': 'case_submitter_id',
                 'as': 'GeneExpression'
             }
         }, {
             '$addFields': {
                 'count': {
                     '$size': '$GeneExpression'
                 }
             }
         }, {
             '$match': {
                 'count': {
                     '$gt': 0
                 }
             }
         }, {
             '$project': {
                 'count': 0
             }
         }

         ])
    async for document in cursor:
        document['BRCA1Carrier'] = len(document['BRCA1']) > 0
        document.pop('BRCA1')
        data.append({'patient': document['case_submitter_id'], 'data': document})

    return data


# @app.get('/clinical_data', response_model=Page[Document], include_in_schema=False)
# async def get_clinical_data(patients: tp.List[str] = Query(None)):
#     return paginate(await query_db(patients))


async def aggregate_db(collection, patients):
    ppln = [
        {
            "$match": {
                'case_submitter_id':
                    {"$in":
                         patients
                     }
            }
        } if patients else
        {"$match":
             {}
         },
        {
            "$group": {
                "_id": "$sample",
                "patient": {
                    "$first": "$patient"
                },
                "names": {
                    "$push": "$name"
                },
                "values": {
                    "$push": "$value"
                }
            }
        },
        {
            "$project": {
                "field": {
                    "$map": {
                        "input": {
                            "$zip": {
                                "inputs": [
                                    "$names",
                                    "$values"
                                ]
                            }
                        },
                        "as": "el",
                        "in": {
                            "name": {
                                "$arrayElemAt": [
                                    "$$el",
                                    0
                                ]
                            },
                            "value": {
                                "$arrayElemAt": [
                                    "$$el",
                                    1
                                ]
                            }
                        }
                    }
                },
                "patient": 1,
                "sample": "$_id",
                "_id": 0
            }
        }
    ]
    db = init_database()
    cursor = db[collection].aggregate(ppln, allowDiskUse=True)
    # return [dict(patient=data['patient'], data=data) for data in cursor]
    async for item in cursor:
        yield orjson.dumps(item).decode() + ',\n'


@app.get('/survival')
async def get_survival(background_task: BackgroundTasks, patients: tp.Tuple[str] = Query(None)):
    return StreamingResponse(aggregate_db('Survival', patients), background=background_task,
                             media_type='application/json')


@app.get('/copy_number')
async def get_survival(background_task: BackgroundTasks, patients: tp.Tuple[str] = Query(None)):
    return StreamingResponse(aggregate_db('CopyNumber', patients), background=background_task,
                             media_type='application/json')


@app.get('/', include_in_schema=False, )
async def docs(request: Request):
    return RedirectResponse(request.scope.get("root_path") + '/docs', status_code=302)


@app.get('/clinical_data')
async def get_clinical_data(background_task: BackgroundTasks, patients: tp.Tuple[str] = Query(None)):
    return StreamingResponse(aggregate_db('ClinicalData', patients), background=background_task,
                             media_type='application/json')


async def _download_dcm_file(link: str, index: int, directory: str):
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(link)
        async with aiofiles.open(Path(directory, f'{index}.dcm'), 'wb') as f:
            await f.write(r.content)


async def _fetch_image_stack(patient: str, sample_number: int):
    patient_samples = (await get_mri_scans(patients=[patient]))[0]
    try:
        patient_sample = patient_samples[sample_number]
    except IndexError:
        raise HTTPException(f'Sample number is out of bounds for {patient}')
    stacks_links = patient_sample['files']
    with tempfile.TemporaryDirectory() as t:
        tasks = [_download_dcm_file(link=link, index=index, directory=t) for index, link in enumerate(stacks_links)]

        await asyncio.gather(*tasks)

        dcm_image = read_dicom_images(t)
    return dcm_image


def _stretch_image(image: np.ndarray) -> np.ndarray:
    image = (255 * (image / np.max(image))).astype(np.uint8)

    image = (skimage.exposure.equalize_adapthist(
        image, kernel_size=15) * 255).astype(np.uint8)
    return image


def _get_mri_patients():
    db = init_database(async_flag=False)
    return db['MRIScans'].find().distinct("patient")


@app.get('/mri_scans', description='Draws the mri scans for the patients '
                                   'from the database. Returns a list of '
                                   'links for each patient. Links are '
                                   'sorted according to slice location')
async def get_mri_scans(patients: tp.Union[tp.List[str]] = Query(None)
                        ):
    ppln = [
        {
            '$project': {
                'patient': 1,
                'sample': 1,
                'value': 1
            }
        }, {
            '$addFields': {
                'temp': {
                    'sample': '$sample',
                    'files': '$value',
                    'patient': '$patient',
                    'series_uid': "$series_uid"
                }
            }
        }, {
            '$group': {
                '_id': '$patient',
                'samples': {
                    '$push': '$temp'
                }
            }
        }, {
            '$project': {
                'patient': '$_id',
                '_id': 0,
                'samples': 1
            }
        }
    ]
    if patients:
        ppln.insert(0, {'$match': {"patient": {"$in": patients}}})
    db = init_database()
    results = await db['MRIScans'].aggregate(ppln).to_list(None)
    for result in results:
        for sample in result['samples']:
            sample['files'] = sorted(sample['files'])
    return [result['samples'] for result in results]


@app.get('/mri_scan_stack', description='Stacks together a single MRI stack for a single patient')
async def get_mri_scan_stack(
        patient: str = Query(default=None, enum=_get_mri_patients()),
        sample_number: int = 0):
    dcm_image = await _fetch_image_stack(patient=patient, sample_number=sample_number)
    return StreamingResponse(BytesIO(pickle.dumps(dcm_image)))


@app.get('/mri_scan_slice')
async def get_mri_scan_slice(
        patient: str = Query(None, enum=_get_mri_patients()),
        sample_number: int = 0,
        slice_number: int = 0):
    dcm_image: np.ndarray = (await _fetch_image_stack(patient=patient, sample_number=sample_number))[slice_number]
    img = _stretch_image(dcm_image)
    _, im_png = cv2.imencode('.png', img)
    return StreamingResponse(BytesIO(im_png.tobytes()), media_type="image/png")


@app.get('/app', include_in_schema=False)
async def get_app(request: Request):
    return {"message": "Hello World", "root_path": request.scope.get("root_path")}


add_pagination(app)
