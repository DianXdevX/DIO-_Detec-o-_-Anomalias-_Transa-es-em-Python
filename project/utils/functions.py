
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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
def apply_regresion_log(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)

    return  model, y_pred, report

def plot_roc_curve(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)

    plt.plot(fpr, tpr)
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    auc = roc_auc_score(y_test, y_probs)

    return auc
def presision_recall (model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_probs)

    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

def apply_undersampling(df):
    frauds = df[df["Class"] == 1]
    normals = df[df["Class"] == 0].sample(
        len(frauds),
        random_state=42
    )

    df_under = pd.concat([frauds, normals])

    return df_under

def apply_smote_oversampling(X, y):
    smote = SMOTE(random_state=42)

    X_res, y_res = smote.fit_resample(X, y)

    return X_res, y_res
def apply_random_forest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)

    report_rf = classification_report(y_test, y_pred_rf)

    return rf, y_pred_rf, report_rf
import matplotlib.pyplot as plt


def save_class_distribution_chart(df, filename, title):
    dist = df["Class"].value_counts()

    plt.figure()
    dist.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_results_txt(
    filename,
    original_shape,
    original_dist,
    under_shape,
    under_dist,
    log_report,
    log_auc,
    rf_report,
    rf_auc,
    initial_chart,
    final_chart
):
    with open(filename, "w", encoding="utf-8") as file:
        file.write("ANOMALY DETECTION PIPELINE RESULTS\n")
        file.write("=" * 40 + "\n\n")

        file.write("ORIGINAL DATASET\n")
        file.write("-" * 40 + "\n")
        file.write(f"Original shape: {original_shape}\n")
        file.write(f"Original class distribution:\n{original_dist}\n\n")
        file.write(f"Initial chart saved as: {initial_chart}\n\n")

        file.write("UNDERSAMPLING DATASET\n")
        file.write("-" * 40 + "\n")
        file.write(f"Undersampling shape: {under_shape}\n")
        file.write(f"Undersampling class distribution:\n{under_dist}\n\n")
        file.write(f"Final chart saved as: {final_chart}\n\n")

        file.write("LOGISTIC REGRESSION RESULTS\n")
        file.write("-" * 40 + "\n")
        file.write(f"Classification report:\n{log_report}\n")
        file.write(f"AUC: {log_auc}\n\n")

        file.write("RANDOM FOREST RESULTS\n")
        file.write("-" * 40 + "\n")
        file.write(f"Classification report:\n{rf_report}\n")
        file.write(f"AUC: {rf_auc}\n\n")

        file.write("Pipeline execution complete.\n")