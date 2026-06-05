import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import (
    load_transaction_data,
    check_class_distribution,
    apply_log_transform,
    apply_regresion_log
    
)

def main():
    print("Initiating Anomaly Detection Pipeline...")

    #load
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = load_transaction_data(url)

    
    print(df.head())
    print(df.shape)
    dist =check_class_distribution(df)
    print(f"Class Distribution:\n{dist}")
    X_train, X_test, y_train, y_test = apply_log_transform(df)

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")




    print("Pipeline execution complete.")




main()  
   