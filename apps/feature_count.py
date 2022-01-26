import streamlit as st
from common.database import init_database
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px



@st.cache(persist=True)
def get_fig_single_col(col):
    db = init_database('brca-reader')
    try:
        items = list(db[col].aggregate([
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
                '$project': {
                    'patient': '$_id', 
                    '_id': 0, 
                    'count': 1
                }
            }
            ], allowDiskUse=True)
            )
            
        df = pd.DataFrame(items)
        return px.histogram(df, x='count', title=col, nbins=100)
    except:
        pass
db = init_database('brca-reader')
for col in tqdm(db.list_collection_names()):
     fig = get_fig_single_col(col)
     st.plotly_chart(fig)