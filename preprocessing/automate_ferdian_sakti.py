import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def load_data(path):
    return pd.read_csv(path)

def clean_string(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def handle_missing(df):
    cat_cols = ['Gender','Married','Dependents','Self_Employed']

    for col in cat_cols:
        df[col].replace('nan', np.nan, inplace=True)
        df[col].fillna(df[col].mode()[0], inplace=True)

    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def handle_outliers(df):
    num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
    z = np.abs(stats.zscore(df[num_cols]))
    return df[(z < 3).all(axis=1)]

def encoding(df):
    df['Gender'] = df['Gender'].str.capitalize()
    df['Married'] = df['Married'].str.capitalize()
    df['Self_Employed'] = df['Self_Employed'].str.capitalize()

    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
    df['Married'] = df['Married'].map({'Yes':1, 'No':0})
    df['Education'] = df['Education'].map({'Graduate':1, 'Not Graduate':0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes':1, 'No':0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    return df

def one_hot(df):
    if 'Property_Area' in df.columns:
        df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)

    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

def feature_engineering(df):
    df['Income_Bin'] = pd.cut(df['ApplicantIncome'], bins=4, labels=False)
    return df

def scaling(df):
    scaler = StandardScaler()
    num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def final_check(df):
    df['Credit_History'] = df['Credit_History'].astype(int)
    assert df.isnull().sum().sum() == 0
    return df

def preprocess(df):
    df = df.copy()

    if 'Loan_ID' in df.columns:
        df.drop(columns=['Loan_ID'], inplace=True)

    df = clean_string(df)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = handle_outliers(df)
    df = encoding(df)
    df = one_hot(df)
    df = feature_engineering(df)
    df = scaling(df)
    df = final_check(df)

    return df

def save_data(df, path):
    df.to_csv(path, index=False)

if __name__ == "__main__":
    input_path = "dataset_raw/loan.csv"
    output_path = "preprocessing/loan_clean.csv"

    df = load_data(input_path)
    df_clean = preprocess(df)
    save_data(df_clean, output_path)

    print("Preprocessing selesai.")