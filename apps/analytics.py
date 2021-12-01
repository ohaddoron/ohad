from src.analytics import plot_overlayed_histogram, two_sided_ks
from src.data_fetching import get_feature_names
import streamlit as st

group1 = '../notebooks/early_neg.csv'
group2 = '../notebooks/late_neg.csv'
feature_name = 'BRCA1'
group1_name = 'early_neg'
group2_name = 'late_neg'
group3 = 'early_pos.csv'
group4 = 'late_neg.csv'
group3_name = '../notebooks/early_pos'
group4_name = '../notebooks/late_pos'

for feature_name in get_feature_names('GeneExpression'):
    fig = plot_overlayed_histogram(group1=group1,
                                   group2=group2,
                                   feature_name=feature_name,
                                   group1_name=group1_name,
                                   group2_name=group2_name,
                                   group3=group3,
                                   group4=group4,
                                   group3_name=group3_name,
                                   group4_name=group4_name)

    st.plotly_chart(fig)

    ks_1_2 = two_sided_ks(group1, group2, feature_name=feature_name)
    ks_3_4 = two_sided_ks(group3, group4, feature_name=feature_name)

    st.caption(f'p_1_2: {ks_1_2.pvalue}; p_3_4: {ks_3_4.pvalue}')
