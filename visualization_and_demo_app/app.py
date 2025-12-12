import streamlit as st
import pandas as pd
import os

from tabs.visualization import render_visualization_tab
from tabs.training import render_training_tab
from utils.preprocess import preprocess_original_csv   # bạn sẽ tạo file này theo hướng dẫn phía dưới

st.set_page_config(
    page_title="PISA Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# 1. Upload CSV
# --------------------------------------------------------------------
st.sidebar.title("📤 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your ORIGINAL CSV file", type=["csv"])

df_imputed = None
name_to_id, id_to_name = None, None

if uploaded_file:
    try:
        df_original = pd.read_csv(uploaded_file)

        # ---------------------
        # TỰ ĐỘNG LÀM SẠCH FILE
        # ---------------------
        df_imputed = preprocess_original_csv(df_original)

        # mapping country
        if 'country' in df_imputed.columns:
            countries = sorted(df_imputed['country'].unique().astype(str))
            name_to_id = {name: i for i, name in enumerate(countries)}
            id_to_name = {i: name for i, name in enumerate(countries)}

        st.success("✅ File uploaded & cleaned successfully!")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")

else:
    st.info("👉 Please upload an ORIGINAL dataset CSV to get started.")


# --------------------------------------------------------------------
# 2. Navigation (only show when data exists)
# --------------------------------------------------------------------
if df_imputed is not None:

    st.sidebar.markdown("---")
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Select Module:",
        ["📊 Data Visualization", "🤖 Model Training"]
    )

    if page == "📊 Data Visualization":
        render_visualization_tab(df_imputed, id_to_name)

    elif page == "🤖 Model Training":
        render_training_tab(df_imputed, name_to_id, id_to_name)
