import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import your custom models
try:
    from models.lightgbm_model import LightGBMModel
    from models.xgboost_model import XgBoostModel
    from metrics.metrics import eval_metrics
except ImportError:
    st.error("Error importing models in training tab.")

def plot_feature_importance_small(model, X, title):
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='#4B4B4B')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Importance', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    return fig

def render_training_tab(df, name_to_id, id_to_name):
    # Prepare Data
    def rating_group(val):
        if val <= 357: return 1
        elif val <= 410: return 2
        elif val <= 442: return 3
        elif val <= 460: return 4
        elif val <= 480: return 5
        elif val <= 508: return 6
        elif val <= 520: return 7
        elif val <= 540: return 8
        else: return 9

    if 'rating_groups' not in df.columns:
        df['rating_groups'] = df['rating'].apply(rating_group)
        
    X = df.drop(columns=['rating', 'rating_groups'])
    y = df['rating']
    stratify_base = df['rating_groups']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=104, test_size=0.06, stratify=stratify_base
    )

    st.sidebar.header("⚙️ Model Hyperparameters")
    n_estimators = st.sidebar.slider("Number of Estimators", 100, 1000, 500, step=50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1)

    # --- Training Section ---
    st.subheader("1. Train & Compare Models")
    
    if 'trained' not in st.session_state:
        st.session_state['trained'] = False

    if st.button("🚀 Train Both Models"):
        with st.spinner("Training models..."):
            # Train XGBoost
            xgb_wrapper = XgBoostModel(model=None, n_estimators=n_estimators, learning_rate=learning_rate)
            xgb_wrapper.fit(X_train, y_train)
            xgb_pred = xgb_wrapper.predict(X_test)
            
            # Train LightGBM
            lgb_wrapper = LightGBMModel(model=None, n_estimators=n_estimators, learning_rate=learning_rate)
            lgb_wrapper.fit(X_train, y_train)
            lgb_pred = lgb_wrapper.predict(X_test)
            
            # Store results
            st.session_state['xgb_model'] = xgb_wrapper
            st.session_state['lgb_model'] = lgb_wrapper
            st.session_state['xgb_metrics'] = eval_metrics(y_test, xgb_pred)
            st.session_state['lgb_metrics'] = eval_metrics(y_test, lgb_pred)
            st.session_state['X_columns'] = X.columns
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['trained'] = True
            
            st.success("Training Complete!")

    # Display Results
    if st.session_state['trained']:
        st.markdown("### 📊 Performance Metrics")
        
        metrics_config = [("MAE", "mae", "inverse"), ("MSE", "mse", "inverse"), ("RMSE", "rmse", "inverse"), ("R2 Score", "r2", "normal")]
        
        for label, key, color_mode in metrics_config:
            xgb_val = st.session_state['xgb_metrics'][key]
            lgb_val = st.session_state['lgb_metrics'][key]
            diff = lgb_val - xgb_val 
            
            m1, m2, m3 = st.columns([1, 2, 2])
            with m1: st.markdown(f"**{label}**")
            with m2: st.metric(label="XGBoost", value=f"{xgb_val:.4f}")
            with m3: st.metric(label="LightGBM", value=f"{lgb_val:.4f}", delta=f"{diff:.4f}", delta_color=color_mode)
            st.divider()

        st.markdown("### 📉 Feature Importance")
        col1, col2 = st.columns(2)
        with col1:
            st.info("XGBoost Importance")
            st.pyplot(plot_feature_importance_small(st.session_state['xgb_model'].model, X, "XGBoost Features"))
        with col2:
            st.info("LightGBM Importance")
            st.pyplot(plot_feature_importance_small(st.session_state['lgb_model'].model, X, "LightGBM Features"))

    # --- Prediction Interfaces ---
    if st.session_state['trained']:
        st.markdown("---")
        tab_live, tab_test = st.tabs(["🔮 Live Prediction", "🔍 Inspect Test Set"])
        
        # Live Prediction
        with tab_live:
            with st.form("prediction_form"):
                cols = st.columns(4) 
                input_data = {}
                skip_cols = ['sex_BOY', 'sex_GIRL', 'sex_TOT']
                sex_choice = st.selectbox("Student Sex Group", ["BOY", "GIRL", "TOT"])
                
                i = 0
                for col_name in st.session_state['X_columns']:
                    if col_name in skip_cols: continue
                    with cols[i % 4]:
                        if col_name == 'country' and name_to_id:
                            c_name = st.selectbox("Country", list(name_to_id.keys()))
                            input_data[col_name] = name_to_id[c_name]
                        else:
                            default_val = float(df[col_name].mean())
                            input_data[col_name] = st.number_input(col_name, value=default_val)
                    i += 1
                
                input_data['sex_BOY'] = 1.0 if sex_choice == "BOY" else 0.0
                input_data['sex_GIRL'] = 1.0 if sex_choice == "GIRL" else 0.0
                input_data['sex_TOT'] = 1.0 if sex_choice == "TOT" else 0.0
                
                if st.form_submit_button("Predict"):
                    input_df = pd.DataFrame([input_data])[st.session_state['X_columns']]
                    p_xgb = st.session_state['xgb_model'].predict(input_df)[0]
                    p_lgb = st.session_state['lgb_model'].predict(input_df)[0]
                    
                    c1, c2 = st.columns(2)
                    c1.metric("XGBoost", f"{p_xgb:.2f}")
                    c2.metric("LightGBM", f"{p_lgb:.2f}")

        # Test Set Inspection
        with tab_test:
            idx = st.slider("Select Test Index", 0, len(st.session_state['X_test'])-1, 0)
            row_X = st.session_state['X_test'].iloc[[idx]].copy() 
            
            actual = st.session_state['y_test'].iloc[idx]
            
            p_xgb = st.session_state['xgb_model'].predict(row_X)[0]
            p_lgb = st.session_state['lgb_model'].predict(row_X)[0]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Actual", f"{actual:.2f}")
            c2.metric("XGBoost", f"{p_xgb:.2f}", delta=f"{p_xgb-actual:.2f}")
            c3.metric("LightGBM", f"{p_lgb:.2f}", delta=f"{p_lgb-actual:.2f}")
            
            if 'country' in row_X.columns and id_to_name:
                row_X['country_name'] = row_X['country'].map(id_to_name)
            st.dataframe(row_X)