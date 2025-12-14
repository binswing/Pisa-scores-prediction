# PISA Score Prediction (Dự đoán chỉ số PISA)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-green)

## Introduction
This project is part of the **Data Engineering (CO3127)** course at **Ho Chi Minh City University of Technology (HCMUT)**.

The goal of this project is to apply Data Science and Machine Learning techniques to predict the **PISA (Programme for International Student Assessment)** scores of various countries. By analyzing macroeconomic, social, and educational data (such as GDP, Gini index, education expenditure, etc.), we aim to quantify the impact of these factors on the quality of education.

The project features a complete pipeline from data cleaning and EDA to model training (XGBoost, LightGBM) and deployment via a **Streamlit Web Application**.

**Google Drive dataset:** [Click here to view the Google Drive](https://drive.google.com/file/d/1FW_Bz784rDFkI-3a9pmERgM47lPhy4j6/view?usp=sharing)


**Github Repo:** [Click here to view the Github Repo](https://github.com/binswing/Pisa-scores-prediction)

**Live Demo:** [Click here to view the App](https://pisa-scores-prediction.streamlit.app)

---

## Team Members (L03 - Team 05)
**Instructor:** Mr. Vũ Ngọc Tú

* **Nguyễn Tuấn Long** - 2311915 (Leader)
* **Đặng Vũ Anh Khoa** - 2311578
* **Nguyễn Thanh Lâm** - 2311824

---

## Dataset
* **Source:** [Kaggle - PISA Results 2000-2022 (Economics and Education) (deleted)](https://www.kaggle.com/datasets/walassetomaz/pisa-results-2000-2022-economics-and-education/data)
* **Scope:** 39 Countries, PISA rating scores from years 2003 to 2018.
* **Key Features:**
    * `expenditure_on_education_pct_gdp`: Investment in education.
    * `gini_index`: Income inequality.
    * `gdp_per_capita_ppp`: Economic strength.
    * `internet_users`: Technological access.
    * `sex`: Gender demographics (Boy, Girl, Total).
    * **Target:** `rating` (PISA Score).

---

## Tech Stack & Methodology

### 1. Data Preprocessing
* **Cleaning:** Dropped columns with excessive missing values (e.g., alcohol consumption).
* **Imputation:** Used **KNN Imputer** ($k=7$) to handle missing values based on sample similarity.
* **Encoding:** One-Hot Encoding for gender; Label Encoding for categorical features.
* **Splitting:** Stratified Train/Test split (94:6) based on rating groups to ensure distribution consistency.

### 2. Machine Learning Models
We implemented and compared two powerful Gradient Boosting algorithms:
* **LightGBM:** Optimized for speed and memory efficiency using leaf-wise growth and histogram-based algorithms.
* **XGBoost:** A robust framework using level-wise growth, regularization ($L_1, L_2$) to prevent overfitting, and parallel processing.

### 3. Application
* **Framework:** Streamlit
* **Features:**
    * Interactive Data Visualization (Correlation Heatmaps, Boxplots, Violin plots).
    * Real-time Model Training & Comparison.
    * Prediction interface for custom input data.

## Installation
- Build
```bash
    #Create venv
    python -m venv .venv
    
    #Activate venv
    source .venv/bin/activate #Linux/Mac
    .venv\Scripts\activate #Windows
    
    #Install requirements
    pip install -r requirements.txt
```
- Run demo app locally
```bash
    cd ./visualization_and_demo_app
    streamlit run app.py
```
- Run PISA predicting code
```bash
    cd ./prediction_code
    python .\pisa_code.py
```