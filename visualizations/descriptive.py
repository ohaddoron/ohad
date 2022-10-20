import csv
from abc import ABC, abstractmethod
from pathlib import Path

import streamlit as st

from src.data_handler import FeaturesMetadataHandler


class AbstractDescriptor(ABC):
    def __init__(self):
        with st.sidebar.container():
            with st.form('modality_selection'):
                modalities = st.multiselect(label='Modalities', options=['miRNA', 'DNAm.yaml', 'mRNA'])
                if not st.form_submit_button():
                    st.stop()

        self.metadata = {modality: FeaturesMetadataHandler(
            mongodb_connection_string='mongodb://TCGAManager:MongoDb-eb954cffde2cedf17b22b@132.66.207.18:80/TCGAOmics?authSource=admin',
            redis_host='localhost',
            db_name='TCGAOmics',
            cache=True,
            redis_port=6379
        ).fetch_features_metadata(modality=modality,
                                  features=list(csv.reader(
                                      Path(__file__).parent.joinpath(f'../resources/{modality}_names.csv').open()))[0])
                         for
                         modality in
                         modalities}

    @abstractmethod
    def render(self):
        ...


class BoxPlotDescriptor(AbstractDescriptor):
    def render(self):
        pass


BoxPlotDescriptor().render()
