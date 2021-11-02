import pprint
from functools import lru_cache

from bson import json_util
from fastapi import FastAPI, Query
from motor import MotorDatabase
from pydantic import BaseModel
import typing as tp
from fastapi_pagination import Page, add_pagination, paginate
from starlette.responses import RedirectResponse, JSONResponse
from aiocache import cached
from common.database import parse_mongodb_connection_string
from src.utils import init_cached_database, get_config
import orjson


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content: tp.Any) -> bytes:
        return orjson.dumps(content)


class Document(BaseModel):
    data: dict


class BSONResponse(JSONResponse):
    media_type = 'application/json'

    def render(self, content: tp.Any) -> bytes:
        return json_util.dumps(content)


# app = FastAPI(title='Fetcher', default_response_class=BSONResponse)
app = FastAPI(title='Fetcher', default_response_class=ORJSONResponse)


@lru_cache
def init_database() -> MotorDatabase:
    config = get_config('omics-database')
    db = init_cached_database(parse_mongodb_connection_string(
        **config), db_name=config['db_name'], async_flag=False)
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


@app.get('/clinical_data', response_model=Page[Document], include_in_schema=False)
async def get_clinical_data(patients: tp.List[str] = Query(None)):
    return paginate(await query_db(patients))


@lru_cache
def aggregate_db(collection, patients):
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
    cursor = db[collection].aggregate(ppln)
    return list(cursor)


@app.get('/survival', response_model=Page[Document])
async def get_survival(patients: tp.List[str] = Query(None)):
    return paginate(aggregate_db('Survival', patients))


@app.get('/', include_in_schema=False)
async def docs():
    return RedirectResponse('/docs')


add_pagination(app)
