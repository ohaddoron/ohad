from functools import lru_cache

import pandas as pd
import pymongo.database
import toml
from pymongo.collection import Collection

from common import config
import os
from loguru import logger
from common.database import parse_mongodb_connection_string, init_cached_database
from PIL import Image
import typing as tp
import numpy as np
import cv2
from tqdm import tqdm
from loguru import logger
from dask import dataframe as dd


@lru_cache
def get_config(name: tp.Optional[str] = None):
    config_path = os.path.join(os.path.dirname(__file__), '../config.toml')
    logger.debug(f'Reading {name} from {config_path}')
    return config.get_config(config_path, name)


def overlay_segmentation_on_image(segmentation_slice: np.array,
                                  image: np.array,
                                  header: tp.List[tp.Tuple[int, int]],
                                  alpha=0.6):
    """
    Overlays a segmentation mask ontop of a dicom image. Performs on a single slice

    :param segmentation_slice: mask image for a single slice
    :type segmentation_slice: np.array
    :param image: dicom image that serves as a base image
    :type image: np.array
    :param header: header for the segmentation map depicting the lesion ROI
    :type header: tp.List[tp.Tuple[int, int]]
    :param alpha: alpha value to use when blending, defaults to 0.4
    :type alpha: float, optional
    :return: overlay of the lesion ontop of the relevant slice
    :rtype: PIL.Image.Image
    """

    image = np.array(Image.fromarray(image).convert('RGB'))

    mask = np.zeros_like(image)

    mask[header[0][0]:header[0][1] + 1, header[1]
                                        [0]:header[1][1] + 1, 1] = segmentation_slice * 255
    image_2 = image.copy()

    image_2 = np.maximum(image_2, mask)

    result = (1 - alpha) * image + alpha * image_2
    return Image.fromarray(result.astype(np.uint8))


def parse_file_to_database(file_name: str,
                           col_name: str,
                           num_rows_to_parse_before_dump: int = 100000,
                           config_name: tp.Optional[str] = 'omics-database',
                           create_index: tp.Optional[bool] = True
                           ):
    """Parses a dataframe from disk into mongodb with the following convention:

    .. code-block:: json
        {
            patient: <patient_name>,
            name: <feature_name>,
            value: <feature_value>
        }

    The function will parse 1e6 entries and insert them to the database rather than inserting each one to conserve time.

    :param col: pymongo collection object to insert data into
    :type col: pymongo.collection.Collection
    :param df: dataframe to read rows from. Should be sorted such that the first item in each row is the feature value
    :type df: dask.dataframe.Dataframe
    :param patients: list of patient identification barcodes
    :type patients: tp.List[str]
    :return:
    """

    config = get_config(config_name)
    db = init_cached_database(parse_mongodb_connection_string(
        **config), db_name=config['db_name'])

    logger.debug(f'Using config: {toml.dumps(config)}')

    df = dd.read_csv(
        filepath_or_buffer=file_name,
        sep='\t'
    )
    patients = df.columns[1:]

    logger.debug(df.head())

    col: Collection = db[col_name]
    col.drop()

    if create_index:
        col.create_index([('patient', 1)])
        col.create_index([('name', 1)])
        col.create_index([('patient', 1), ('name', 1)], unique='chr' not in df.columns.to_list())

        logger.debug(f'Collection indexes: {col.index_information()}')
    aggregator = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        assert isinstance(row[0], str), 'first item in each row must be the feature name'
        for patient, value in zip(patients, row[1:]):
            if 'chr' in df.columns.to_list():
                pass
            else:
                assert isinstance(value, (int, float)), f'Values must be floating point objects, got instead: {value}'

            aggregator.append({"patient": patient, "name": row[0], "value": value})
            if (len(aggregator) % num_rows_to_parse_before_dump) == 0:
                col.insert_many(aggregator, bypass_document_validation=True)
                aggregator = []
    if aggregator:
        col.insert_many(aggregator, bypass_document_validation=True)
