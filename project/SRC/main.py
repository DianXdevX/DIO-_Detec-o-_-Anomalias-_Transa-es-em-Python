import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import (
    load_transaction_data,
    check_class_distribution,
    apply_log_transform,
    apply_regresion_log,
    plot_roc_curve,
    presision_recall,
    apply_undersampling,
    apply_random_forest,
    save_class_distribution_chart,
    save_results_txt
)


def main():
    print("Initiating Anomaly Detection Pipeline...")

    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = load_transaction_data(url)

    print("Original dataset:")
    print(df.head())
    print(df.shape)

    # original dataset info
    original_shape = df.shape
    original_dist = check_class_distribution(df)

    print(f"Original Class Distribution:\n{original_dist}")

    # save initial chart
    initial_chart = "class_distribution_original.png"
    save_class_distribution_chart(
        df,
        initial_chart,
        "Original Class Distribution"
    )

    # create copy and apply undersampling
    df_under = apply_undersampling(df.copy())

    print("Shape after undersampling:")
    print(df_under.shape)

    # undersampled dataset info
    under_shape = df_under.shape
    under_dist = check_class_distribution(df_under)

    print(f"Class Distribution after undersampling:\n{under_dist}")

    # save final chart
    final_chart = "class_distribution_under.png"
    save_class_distribution_chart(
        df_under,
        final_chart,
        "Class Distribution After Undersampling"
    )

    # use only the undersampled copy from here
    X_train, X_test, y_train, y_test = apply_log_transform(df_under)

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Logistic Regression
    model, y_pred, report = apply_regresion_log(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(f"Logistic Regression Classification Report:\n{report}")

    auc = plot_roc_curve(model, X_test, y_test)
    print(f"Logistic Regression AUC: {auc}")

    presision_recall(model, X_test, y_test)

    # Random Forest
    rf_model, y_pred_rf, report_rf = apply_random_forest(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(f"Random Forest Classification Report:\n{report_rf}")

    auc_rf = plot_roc_curve(rf_model, X_test, y_test)
    print(f"Random Forest AUC: {auc_rf}")

    presision_recall(rf_model, X_test, y_test)

    # save txt results
    save_results_txt(
        filename="results.txt",
        original_shape=original_shape,
        original_dist=original_dist,
        under_shape=under_shape,
        under_dist=under_dist,
        log_report=report,
        log_auc=auc,
        rf_report=report_rf,
        rf_auc=auc_rf,
        initial_chart=initial_chart,
        final_chart=final_chart
    )

    print("Results saved in results.txt")
    print("Pipeline execution complete.")


main()