# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import KNNImputer

# def preprocess_original_csv(df):
#     df = df.copy()

#     if 'country' in df.columns:
#         df['country_name'] = df['country'].astype(str)
#     else:
#         raise ValueError("❌ ORIGINAL CSV missing 'country' column")

#     # 1. Drop unused columns
#     drop_cols = ['index_code', 'alcohol_consumption_per_capita']
#     df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

#     # 2. One hot encode sex
#     if 'sex' in df.columns:
#         df = pd.get_dummies(df, columns=['sex'], prefix='sex')

#     # 3. Label Encode remaining object columns
#     le = LabelEncoder()
#     for col in df.select_dtypes(include='object').columns:
#         df[col] = le.fit_transform(df[col].astype(str))

#     # 4. Impute missing values
#     imputer = KNNImputer(n_neighbors=7)
#     df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

#     return df

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

def preprocess_original_csv(df):
    df = df.copy()

    # Keep country_name as ISO-3 string
    if "country" in df.columns:
        df["country_name"] = df["country"].astype(str)
        df = df.drop(columns=["country"])
    else:
        raise ValueError("❌ ORIGINAL CSV missing 'country' column")

    # Drop unused
    drop_cols = ["index_code", "alcohol_consumption_per_capita"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # One-hot encode sex
    if "sex" in df.columns:
        df = pd.get_dummies(df, columns=["sex"], prefix="sex")

    # Label encode ONLY object columns EXCEPT country_name
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        if col != "country_name":       # DO NOT ENCODE country_name
            df[col] = le.fit_transform(df[col].astype(str))

    # Impute numeric values only
    numeric_cols = [c for c in df.columns if c != "country_name"]
    imputer = KNNImputer(n_neighbors=7)
    df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

    # Combine back country_name
    df_final = pd.concat([df_numeric, df["country_name"]], axis=1)

    return df_final
