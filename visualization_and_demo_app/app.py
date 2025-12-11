import streamlit as st
import pandas as pd
import os
from tabs.visualization import render_visualization_tab
from tabs.training import render_training_tab

st.set_page_config(page_title="PISA Analytics", layout="wide", initial_sidebar_state="expanded")
@st.cache_data
def load_data():
    def robust_read_csv(filename):
        paths = [filename, f"../dataset/{filename}", f"dataset/{filename}"]
        for p in paths:
            if os.path.exists(p): return pd.read_csv(p)
        return None

    df_imputed = robust_read_csv("economics_and_education_dataset_CSV_imputed.csv")
    df_original = robust_read_csv("economics_and_education_dataset_CSV.csv")
    
    if df_imputed is None:
        st.error("❌ Data not found! Please check your dataset folder.")
        return None, None, None
    name_to_id, id_to_name = None, None
    if df_original is not None and 'country' in df_original.columns:
        countries = sorted(df_original['country'].unique().astype(str))
        name_to_id = {name: i for i, name in enumerate(countries)}
        id_to_name = {i: name for i, name in enumerate(countries)}
        
    return df_imputed, name_to_id, id_to_name

df, name_to_id, id_to_name = load_data()
if df is not None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module:", 
        ["Data Visualization", "Model Training"]
    )
    
    st.sidebar.markdown("---")
    if page == "Data Visualization":
        render_visualization_tab(df, id_to_name)
    elif page == "Model Training":
        render_training_tab(df, name_to_id, id_to_name)