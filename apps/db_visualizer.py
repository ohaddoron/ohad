import numpy as np
import pandas as pd
import pymongo
import streamlit as st
import plotly.express as px
from tqdm import tqdm

from common.database import init_database
from src.logger import logger

CONFIG_NAME = 'omicsdb'


class VisualizerBase:
    def __init__(self, collection: str, config_name: str):
        self.collection_name = collection
        self.config_name = config_name


class FeatureTypesVisualizer(VisualizerBase):

    @st.cache(persist=True)
    def get_visualization_data(self):
        db = init_database(config_name=self.config_name)
        cursor = db[self.collection_name].aggregate([
            {
                '$group': {
                    '_id': '$patient',
                    'names': {
                        '$push': '$name'
                    }
                }
            }, {
                '$addFields': {
                    'count': {
                        '$size': '$names'
                    }
                }
            }, {
                '$group': {
                    '_id': '$names',
                    'patients': {
                        '$push': '$_id'
                    }
                }
            }, {
                '$project': {
                    'patient_count': {
                        '$size': '$patients'
                    },
                    'names_count': {
                        '$size': '$_id'
                    },
                    'names': '$_id',
                    'patients': 1,
                    '_id': 0
                }
            }
        ], allowDiskUse=True)

        if not cursor.alive:
            return
        items = list(cursor)
        df = pd.DataFrame(sorted(items, key=lambda x: x['names_count']))
        cumsum = np.cumsum(df['patient_count'])
        df_ = pd.DataFrame(
            {'Subset#': list(np.arange(1, len(cumsum) + 1)), "Patients": cumsum / list(cumsum)[-1],
             "Feature Count": df['names_count'].values})
        return dict(df=df_, title=self.collection_name, caption=f'Total number of patients: {list(cumsum)[-1]}')

    @classmethod
    def visualize(cls, collection: str, config_name: str):
        obj = cls(collection, config_name)

        viz_data = obj.get_visualization_data()
        if not viz_data:
            st.info(f'{collection} is unavailable')

        fig = px.line(viz_data['df'], x='Subset#', y="Patients", markers=True, title=viz_data['title'])

        st.plotly_chart(fig)
        st.caption(viz_data['caption'])


def main():
    db = init_database(CONFIG_NAME)
    for col in tqdm(sorted(db.list_collection_names())):
        try:
            st.info(col)
            FeatureTypesVisualizer.visualize(col, config_name=CONFIG_NAME)
        except:
            st.error(col)
            continue


main()
