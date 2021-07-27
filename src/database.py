# Created by ohad at 7/27/21
import typing as tp

from loguru import logger

from common.database import connect_to_database


def get_segmentation_files(bcr_patient_barcode: str, db_config: dict) -> tp.List[str]:
    """
    Fetches all segmentation files for a specific patient

    >>> get_segmentation_files('TCGA-AO-A12D')

    :param bcr_patient_barcode: patient barcode id
    :type bcr_patient_barcode: str
    :param db_config: database configuration dictionary, defaults to None
    :type db_config: tp.Optional[str], optional
    :return: list of segmentation file paths
    :rtype: List[str]
    """
    db = connect_to_database(db_config=db_config)

    return [item['segmentation_file'] for item in
            db['segmentation_files'].find({'bcr_patient_barcode': bcr_patient_barcode})]


def get_dcm_dirs(bcr_patient_barcode: str, db_config: dict) -> tp.List[str]:
    """
    Fetches all dicom directories for a specific patient

    >>> get_dcm_dirs('TCGA-AO-A12D')

    :param bcr_patient_barcode: patient barcode id
    :type bcr_patient_barcode: str
    :param db_config: database configuration dictionary, defaults to None
    :type db_config: tp.Optional[str], optional
    :return: list of dicom directories paths
    :rtype: List[str]
    """
    db = connect_to_database(db_config)
    return [item['dcm_dir'] for item in (db['dcm_files'].find({'bcr_patient_barcode': bcr_patient_barcode}))]


def get_unique_patient_barcodes(collection_name: str, db_config: dict) -> tp.List[str]:
    """
    Fetches unique patient barcode values for a specific collection

    :param collection_name: collection name to query
    :type collection_name: str
    :param db_config: database configuration dictionary, defaults to None
    :type db_config: tp.Optional[str], optional
    :return: list of unique barcode names
    :rtype: tp.List[str]
    """
    db = connect_to_database(db_config)
    col = db[collection_name]
    return col.find({}).distinct("bcr_patient_barcode")


def get_series_uids(collection_name: str, patient_barcode: str, db_config: dict) -> tp.List[str]:
    """
    Fetches uids for a specific patient barcode

    :param collection_name: collection name to query
    :type collection_name: str
    :param patient_barcode: [unique identifier for a patient
    :type patient_barcode: str
    :return: series uids matching specific dicom images
    :param db_config: database configuration dictionary, defaults to None
    :type db_config: tp.Optional[str], optional
    :rtype: tp.List[str]
    """
    db = connect_to_database(db_config)
    col = db[collection_name]
    return col.find({"bcr_patient_barcode": patient_barcode}).distinct("series_uid")
