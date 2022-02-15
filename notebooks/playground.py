# from pymongo import MongoClient
# import dask.bag as db
# import pandas as pd
#
# # Requires the PyMongo package.
# # https://api.mongodb.com/python/current
#
# client = MongoClient(
#     'mongodb://ohad:MongoDb-2c29fdd180dfc2f6f423a@132.66.207.18:80/?authSource=admin&authMechanism=SCRAM-SHA-256&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
# result = client['brca_omics']['CopyNumber'].aggregate([
#     {
#         '$group': {
#             '_id': '$sample',
#             'data': {
#                 '$push': {
#                     'k': '$name',
#                     'v': '$value'
#                 }
#             },
#             'patient': {
#                 '$push': '$patient'
#             }
#         }
#     }, {
#         '$project': {
#             'user': {
#                 '$arrayElemAt': [
#                     '$patient', 0
#                 ]
#             },
#             'sample': '$_id',
#             '_id': 0,
#             'data': {
#                 '$arrayToObject': '$data'
#             }
#         }
#     }, {
#         '$replaceRoot': {
#             'newRoot': {
#                 '$mergeObjects': [
#                     '$$ROOT', '$data'
#                 ]
#             }
#         }
#     }, {
#         '$project': {
#             'data': 0
#         }
#     }
# ],
#     allowDiskUse=True
# )
#
# if __name__ == '__main__':
#     df = pd.DataFrame(result)
#
#     a = 1
import asyncio

from motor import MotorCollection

from common.database import init_database
from motor.motor_asyncio import AsyncIOMotorDatabase

from src.logger import logger

db = init_database(config_name='omics-database')


def add_index_to_col(col_name, config_name: str = 'omics-database'):
    logger.debug(f'Working on: {col_name}')
    db = init_database(config_name, async_flag=False)
    col = db[col_name]
    col.create_index("patient", background=True)
    col.create_index("name", background=True)
    col.create_index([("patient", 1), ("name", 1)], background=True)
    col.create_index([("sample", 1), ("name", 1)], background=True)
    col.create_index([("sample", 1), ("patient", 1)], background=True)


def main():
    for col_name in db.list_collection_names():
        add_index_to_col(col_name)


if __name__ == '__main__':
    main()
