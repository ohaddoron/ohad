from common import config
import os
from loguru import logger
from common.database import parse_mongodb_connection_string, init_cached_database
from PIL import Image
import typing as tp
import numpy as np
import cv2


def get_config(name: tp.Optional[str] = None):
    config_path = os.path.join(os.path.dirname(__file__), '../config.toml')
    logger.debug(f'Reading {name} from {config_path}')
    return config.get_config(config_path, name)


def connect_to_database(config_name: tp.Optional[str] = None):
    config_name = config_name or 'brca-database'
    conf = get_config(config_name)
    return init_cached_database(parse_mongodb_connection_string(
        **conf), db_name=conf["db_name"])


def get_segmentation_files(bcr_patient_barcode: str, config_name: tp.Optional[str] = None):
    config_name = config_name or 'brca-database'
    db = connect_to_database(config_name=config_name)

    return [item['segmentation_file'] for item in
            db['segmentation_files'].find({'bcr_patient_barcode': bcr_patient_barcode})]


def get_dcm_dirs(bcr_patient_barcode: str, config_name: tp.Optional[str] = None):
    config_name = config_name or 'brca-database'
    db = connect_to_database(config_name=config_name)

    return [item['dcm_dir'] for item in (db['dcm_files'].find({'bcr_patient_barcode': bcr_patient_barcode}))]


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
