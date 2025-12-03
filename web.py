import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import xgboost as xgb
import lightgbm as lgb

st.set_page_config(page_title="PISA ML & Visualization", layout="wide")

@st.cache_data
def load_data():
    #sửa cái này theo đường dẫn dataset
    path = "economics_and_education_dataset_CSV_clean.csv"
    return pd.read_csv(path)

df = load_data()

st.title("PISA Dataset – Visualization & Machine Learning")
st.write("Web phục vụ nghiên cứu **mô hình hồi quy dự đoán Chỉ số PISA** và phân tích ảnh hưởng của các biến kinh tế – xã hội – giới tính.")

# TABS GIAO DIỆN
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tổng quan dữ liệu",
    "Phân tích theo quốc gia",
    "Phân tích theo giới tính",
    "Mô hình dự đoán PISA",
    "So sánh mô hình XGBoost và LightGBM"
])

# TAB 1 — Tổng quan dữ liệu
with tab1:
    st.header("Tổng quan dữ liệu")
    st.dataframe(df.head())
    
    st.subheader("Thống kê mô tả")
    st.dataframe(df.describe())

    st.subheader("Các cột dữ liệu có trong tập")
    st.write(df.columns.tolist())

# TAB 2 — Phân tích theo quốc gia
with tab2:
    st.header("Visualize by Country")
        # Danh sách quốc gia
    country_list = sorted(df["country"].dropna().unique())
    selected_country = st.selectbox("Chọn quốc gia:", country_list)
        # Lọc dữ liệu
    df_country = df[df["country"] == selected_country]
    st.subheader(f"Dữ liệu cho: **{selected_country}**")
    st.dataframe(df_country)
        # --- Plot theo thời gian ---
    st.subheader("Điểm PISA theo thời gian")
        # Dataset có 'rating'
    metric = "rating"
    fig, ax = plt.subplots()
    ax.plot(df_country["time"], df_country[metric], marker="o")
    ax.set_xlabel("Năm")
    ax.set_ylabel("PISA Rating")
    ax.set_title(f"PISA Rating của {selected_country} theo thời gian")
    fig.tight_layout()
    st.pyplot(fig)


# TAB 3 — Phân tích theo giới tính
with tab3:
    st.header("Phân tích giới tính")
    gender_score_cols = ["rating"]
    selected_metric = st.selectbox("Chọn loại điểm:", gender_score_cols)
    df_gender = df.groupby("sex")[selected_metric].mean()
    st.subheader(f"Điểm trung bình theo giới tính ({selected_metric})")
    fig, ax = plt.subplots()
    ax.bar(df_gender.index, df_gender.values)
    ax.set_ylabel(selected_metric)
    ax.set_title("So sánh điểm theo giới tính")
    fig.tight_layout()
    st.pyplot(fig)

# TAB 5 — XGBoost & LightGBM
with tab5:
    st.header("So sánh mô hình XGBoost và LightGBM")
    st.write("Dữ liệu được dùng từ tập đã làm sạch. Model sử dụng biến mục tiêu: **rating**")
    # --- Chuẩn bị dữ liệu ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols].drop(columns=["rating"]).fillna(0)
    y = df["rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # 1) XGBoost
    st.subheader("XGBoost Regression")
    xgbmodel = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.1,
        max_depth=8,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42
    )

    xgbmodel.fit(X_train, y_train)
    xgb_pred = xgbmodel.predict(X_test)

    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = mean_squared_error(y_test, xgb_pred) ** 0.5

    st.write(f"**MAE:** {xgb_mae:.3f}")
    st.write(f"**RMSE:** {xgb_rmse:.3f}")

    # Feature importance XGBoost
    st.write("### Feature Importance (XGBoost)")
    fig, ax = plt.subplots(figsize=(6, 6))
    importance_xgb = xgbmodel.feature_importances_
    sorted_idx = np.argsort(importance_xgb)

    ax.barh(X.columns[sorted_idx], importance_xgb[sorted_idx])
    ax.set_title("XGBoost Feature Importance")
    st.pyplot(fig)

    # 2) LightGBM
    st.subheader("LightGBM Regression")

    lgbmodel = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.1,
        max_depth=12,
        num_leaves=16,
        min_child_samples=10,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42
    )

    lgbmodel.fit(X_train, y_train)
    lgb_pred = lgbmodel.predict(X_test)

    lgb_mae = mean_absolute_error(y_test, lgb_pred)
    lgb_rmse = mean_squared_error(y_test, lgb_pred) ** 0.5

    st.write(f"**MAE:** {lgb_mae:.3f}")
    st.write(f"**RMSE:** {lgb_rmse:.3f}")

    # Feature importance LightGBM
    st.write("### Feature Importance (LightGBM)")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    importance_lgb = lgbmodel.feature_importances_
    sorted_idx_2 = np.argsort(importance_lgb)

    ax2.barh(X.columns[sorted_idx_2], importance_lgb[sorted_idx_2])
    ax2.set_title("LightGBM Feature Importance")
    st.pyplot(fig2)

    # 3) So sánh mô hình
    st.subheader("So sánh XGBoost vs LightGBM")

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.bar(["XGBoost-MAE", "LightGBM-MAE"], [xgb_mae, lgb_mae])
    ax3.set_title("So sánh MAE giữa 2 mô hình")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.bar(["XGBoost-RMSE", "LightGBM-RMSE"], [xgb_rmse, lgb_rmse])
    ax4.set_title("So sánh RMSE giữa 2 mô hình")
    st.pyplot(fig4)


    #run local: streamlit run web.py