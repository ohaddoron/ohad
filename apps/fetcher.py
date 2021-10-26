import pprint
from functools import lru_cache

from fastapi import FastAPI, Query
from motor import MotorDatabase
from pydantic import BaseModel
import typing as tp
from fastapi_pagination import Page, add_pagination, paginate
from starlette.responses import RedirectResponse
from aiocache import cached
from common.database import parse_mongodb_connection_string
from src.utils import init_cached_database, get_config


class Document(BaseModel):
    patient: str
    data: dict


app = FastAPI(title='Fetcher')


@lru_cache
def init_database() -> MotorDatabase:
    config = get_config('omics-database')
    db = init_cached_database(parse_mongodb_connection_string(
        **config), db_name=config['db_name'], async_flag=True)
    return db


@cached(ttl=600)
async def query_db(patients: tp.List[str]):
    db = init_database()
    data = []
    cursor = db['ClinicalData'].aggregate(
        [{"$match": {'case_submitter_id': {"$in": patients}}} if patients else {"$match": {}},

         {
             '$addFields': {
                 'patient': '$case_id'
             }
         }, {
             '$lookup': {
                 'from': 'ClinicalBRCA1',
                 'localField': 'patient',
                 'foreignField': 'case_submitter_id',
                 'as': 'BRCA1'
             }
         }, {
             '$lookup': {
                 'from': 'ExonExpression',
                 'localField': 'patient',
                 'foreignField': 'patient',
                 'as': 'ExonExpression'
             }
         }

         ])
    async for document in cursor:
        document.pop('_id')
        document['BRCA1Carrier'] = len(document['BRCA1']) > 0
        document.pop('BRCA1')
        data.append({'patient': document['case_submitter_id'], 'data': document})

    return data


@app.get('/clinical_data', response_model=Page[Document])
async def get_clinical_data(patients: tp.List[str] = Query(None)):
    return paginate(await query_db(patients))


@app.get('/', include_in_schema=False)
async def docs():
    return RedirectResponse('/docs')


add_pagination(app)
