import dask.dataframe as dd
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import plotly.express as px


def two_sided_ks(group1: str, group2: str, feature_name: str):
    group1_df = dd.read_csv(group1)
    group2_df = dd.read_csv(group2)

    return ks_2samp(group1_df.groupby('name').get_group(feature_name).value,
                    group2_df.groupby('name').get_group(feature_name).value)


def plot_overlayed_histogram(group1: str,
                             group2: str,
                             feature_name: str,
                             group1_name: str = None,
                             group2_name: str = None,
                             group3: str = None,
                             group4: str = None,
                             group3_name: str = None,
                             group4_name: str = None):
    group1_df = dd.read_csv(group1)
    group2_df = dd.read_csv(group2)

    group1_data = group1_df.groupby('name').get_group(feature_name).value.compute()
    group2_data = group2_df.groupby('name').get_group(feature_name).value.compute()
    data = np.concatenate((group1_data, group2_data))
    series = [f'{group1_name}_{feature_name}'] * len(group1_data) + [f'{group2_name}_{feature_name}'] * len(group2_data)

    if group3 is not None and group4 is not None:
        group3_df = dd.read_csv(group1)
        group4_df = dd.read_csv(group2)

        group3_data = group3_df.groupby('name').get_group(feature_name).value.compute()
        group4_data = group4_df.groupby('name').get_group(feature_name).value.compute()

        data = np.concatenate((data, np.concatenate((group3_data, group4_data))))
        series = series + [f'{group3_name}_{feature_name}'] * len(group3_data) + [
            f'{group4_name}_{feature_name}'] * len(
            group4_data)

    df = pd.DataFrame(dict(data=data,
                           series=series))

    return px.histogram(df, x='data',
                        color='series', nbins=100, opacity=0.75)
