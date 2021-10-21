import SimpleITK as sitk
import numpy as np
import skimage
import skimage.exposure
import streamlit as st
from common.les_files import read_all_maps_from_les_file
from common.utils import read_dicom_images

from src.database import get_segmentation_files, get_dcm_dirs, get_unique_patient_barcodes, get_series_uids
from src.utils import overlay_segmentation_on_image, get_config
from common.database import connect_to_database

db_config = get_config('brca-database')

@st.cache(ttl=600)
def get_unique_barcode_names():
    patient_barcodes = get_unique_patient_barcodes(collection_name='segmentation_files', db_config=db_config)
    return patient_barcodes


@st.cache(ttl=600)
def get_series_uids_for_display(patient_barcode):
    series_uids = get_series_uids(collection_name='tcga_breast_radiologist_reads', patient_barcode=patient_barcode, db_config=db_config)
    return series_uids


def main():
    with st.sidebar.beta_container():
        patient_barcode = st.selectbox(
            label='Patient barcode', options=get_unique_barcode_names())

        segmentation_file = get_segmentation_files(
            bcr_patient_barcode=patient_barcode, db_config=db_config)[0]

        series_uids = get_series_uids_for_display(patient_barcode)
        dcm_dirs = get_dcm_dirs(bcr_patient_barcode=patient_barcode, db_config=db_config)

        dirs_to_use = []
        for series_uid in series_uids:
            dirs_to_use.append(
                [dcm_dir for dcm_dir in dcm_dirs if series_uid in dcm_dir][0])
        dcm_dirs = dirs_to_use

        segmentation_maps = read_all_maps_from_les_file(
            segmentation_file)

        slice_range = segmentation_maps[0]['header']

        segmentation_slices = (segmentation_maps[0]['data']).astype(np.bool8)

        images_to_display = st.multiselect(
            label='Dicom Images', options=dcm_dirs, default=dcm_dirs)

        alpha = st.slider(label='Alpha value', min_value=0., max_value=1., value=0.6)

    with st.beta_container():
        for dicom_dir in images_to_display:
            st.header(dicom_dir.split('/')[-1])

            images = st.cache(read_dicom_images)(dicom_dir)
            for image, segmentation_slice in zip(images[slice_range[2][0]:slice_range[2][1] + 1], segmentation_slices):
                cols = st.beta_columns(2)

                image = (255 * (image / np.max(image))).astype(np.uint8)

                image = (skimage.exposure.equalize_adapthist(
                    image, kernel_size=15) * 255).astype(np.uint8)
                cols[0].image(image)

                cols[1].image(overlay_segmentation_on_image(
                    segmentation_slice=segmentation_slice, image=image, header=slice_range, alpha=alpha))


if __name__ == '__main__':
    main()
