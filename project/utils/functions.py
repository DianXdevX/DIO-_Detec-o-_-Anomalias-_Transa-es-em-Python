
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_transaction_data(url:str):
    #url= "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    return pd.read_csv(url)

def check_class_distribution(df: pd.DataFrame) :
    return df["Class"].value_counts(normalize=True)

def apply_log_transform(df) :
    df["Amount_log"] = np.log1p(df["Amount"])

    scaler = StandardScaler()
    features_to_scale = ["Amount_log", "Time"]
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.3,
        random_state=42
    )

    return  X_train, X_test, y_train, y_test
def apply_regresion_log(df):
    
