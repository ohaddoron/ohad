import pandas as pd
import streamlit as st
from pymongo.collection import Collection
from plotly import graph_objects as go
from common.database import parse_mongodb_connection_string
from src.utils import init_cached_database, get_config
from loguru import logger

config = get_config('omics-database')
db = init_cached_database(parse_mongodb_connection_string(
    **config), db_name=config['db_name'])

cols = db.list_collection_names()
cols.sort()


# col_name = st.sidebar.selectbox(label='Collection Name', options=cols, index=0)


@st.cache(persist=True)
def histogram_aggregation(col_name: str, query: dict = None):
    db = init_cached_database(parse_mongodb_connection_string(
        **config), db_name=config['db_name'])
    col = db[col_name]
    ppln = [
        {
            '$bucketAuto': {
                'groupBy': '$value',
                'buckets': 1500
            }
        }, {
            '$project': {
                'bin_center': {
                    '$divide': [
                        {
                            '$sum': [
                                '$_id.max', '$_id.min'
                            ]
                        }, 2
                    ]
                },
                'count': 1,
                '_id': 0
            }
        }
    ]
    if query:
        ppln.insert(0, query)
    return pd.DataFrame(col.aggregate(ppln, allowDiskUse=True))


@st.cache(persist=True)
def get_unique_names(col_name):
    db = init_cached_database(parse_mongodb_connection_string(
        **config), db_name=config['db_name'], async_flag=False)
    col = db[col_name]

    names = col.distinct("name")
    names.sort()
    return names


@st.cache(persist=True)
def get_histogram_data(col_name: str):
    db = init_cached_database(parse_mongodb_connection_string(
        **config), db_name=config['db_name'], async_flag=False)
    col = db[col_name]

    names = col.distinct("name")
    with st.sidebar.container():
        with st.form(key=f'{col_name}-feature-form'):
            selected_names = st.multiselect('Feature Name', options=names, key=f'{col_name}-feature')
            st.form_submit_button()

    return [histogram_aggregation(col_name=col.name, query=None)] + [
        histogram_aggregation(col_name=col.name, query={"$match": {"name": name}}) for name in selected_names]


@st.cache(persist=True)
def get_patients_count(col_name):
    db = init_cached_database(parse_mongodb_connection_string(
        **config), db_name=config['db_name'], async_flag=False)
    col = db[col_name]
    ppln = [
        {
            '$group': {
                '_id': '$patient',
                'count': {
                    '$sum': 1
                }
            }
        }, {
            '$sort': {
                'count': 1
            }
        }, {
            '$project': {
                'id': '$_id',
                '_id': 0,
                'count': 1
            }
        }
    ]
    return pd.DataFrame(col.aggregate(ppln))


# dfs = get_histogram_data(col_name)
#
# for df in dfs:
#     fig = go.Figure(data=go.Scatter(x=df['bin_center'], y=df['count'] / sum(df['count'])))
#     st.plotly_chart(fig)


for col in cols:
    st.header(col)
    feature_names = get_unique_names(col_name=col)
    with st.expander('Feature Names'):
        st.write(feature_names)

    try:
        patient_counts = get_patients_count(col_name=col)
        with st.expander('Patient Distribution'):
            fig = go.Figure(
                data=go.Bar(x=patient_counts['id'], y=patient_counts['count'] / sum(patient_counts['count'])))
            st.plotly_chart(fig)
    except:
        pass

    try:
        histogram_data = histogram_aggregation(col_name=col)
        with st.expander('Value Distribution'):
            fig = go.Figure(
                data=go.Bar(x=histogram_data['bin_center'], y=histogram_data['count'] / sum(histogram_data['count']),
                            width=0.01))
            st.plotly_chart(fig)
    except:
        pass
