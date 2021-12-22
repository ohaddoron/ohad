from pymongo import MongoClient
import dask.bag as db
import pandas as pd

# Requires the PyMongo package.
# https://api.mongodb.com/python/current

client = MongoClient(
    'mongodb://ohad:MongoDb-2c29fdd180dfc2f6f423a@132.66.207.18:80/?authSource=admin&authMechanism=SCRAM-SHA-256&readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false')
result = client['brca_omics']['CopyNumber'].aggregate([
    {
        '$group': {
            '_id': '$sample',
            'data': {
                '$push': {
                    'k': '$name',
                    'v': '$value'
                }
            },
            'patient': {
                '$push': '$patient'
            }
        }
    }, {
        '$project': {
            'user': {
                '$arrayElemAt': [
                    '$patient', 0
                ]
            },
            'sample': '$_id',
            '_id': 0,
            'data': {
                '$arrayToObject': '$data'
            }
        }
    }, {
        '$replaceRoot': {
            'newRoot': {
                '$mergeObjects': [
                    '$$ROOT', '$data'
                ]
            }
        }
    }, {
        '$project': {
            'data': 0
        }
    }
],
    allowDiskUse=True
)

if __name__ == '__main__':
    df = pd.DataFrame(result)

    a = 1
