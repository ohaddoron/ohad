from common.utils import init_cached_database
import streamlit as st
import SimpleITK as sitk
from common.les_files import read_all_maps_from_les_file
import numpy as np
from glob import glob
from loguru import logger
import os
import sys
import pathlib
from src.utils import connect_to_database, get_dcm_dirs, get_segmentation_files
from common.utils import get_unique_patient_barcodes


@st.cache
def read_dicom_images(dicom_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    reader = sitk.ImageSeriesReader()
    nda = sitk.GetArrayFromImage(image)
    return nda


@st.cache(ttl=600)
def get_unique_barcode_names():
    db = connect_to_database()
    patient_barcodes = get_unique_patient_barcodes(db['segmentation_files'])
    return patient_barcodes


with st.sidebar.beta_container():
    patient_barcode = st.selectbox(
        label='Patient barcode', options=get_unique_barcode_names())

    segmentation_file = get_segmentation_files(
        bcr_patient_barcode=patient_barcode)[0]
    dcm_dirs = get_dcm_dirs(bcr_patient_barcode=patient_barcode)

    segmentation_maps = read_all_maps_from_les_file(
        segmentation_file)

    slice_range = segmentation_maps[0]['header']

    segmentation_slices = (segmentation_maps[0]['data']).astype(np.bool8)

    image_index = st.slider(
        label='Image index', min_value=0, max_value=len(dcm_dirs))

    dicom_dir = dcm_dirs[image_index]


with st.beta_container():
    st.header(dicom_dir.split('/')[-1])
    # logger.critical(dicom_dir)
    images = read_dicom_images(dicom_dir)
    for image, segmentation_slice in zip(images[slice_range[2][0]:slice_range[2][1] + 1], segmentation_slices):
        cols = st.beta_columns(3)

        cols[0].image(image, clamp=True)
        image = (255 * (image / np.max(image))).astype(np.uint8)
        cols[1].image(image)
        cols[2].image((255 * segmentation_slice).astype(np.uint8))
