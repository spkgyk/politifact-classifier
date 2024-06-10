from sklearn.metrics import (
    classification_report,
    matthews_corrcoef,
    confusion_matrix,
    accuracy_score,
)
import pandas as pd


def calculate_metrics(predictions, references):
    report = classification_report(references, predictions, output_dict=True)
    conf_matrix = confusion_matrix(references, predictions)
    mcc = matthews_corrcoef(references, predictions)
    acc = accuracy_score(references, predictions)

    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop("accuracy")

    macro_avg = report_df.loc["macro avg"]
    precision, recall, f1_score = macro_avg[["precision", "recall", "f1-score"]]

    conf_matrix_df = format_confusion_matrix(conf_matrix)

    training_metrics = {
        "accuracy": acc,
        "matthews_correlation": mcc,
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score,
    }

    return training_metrics, conf_matrix_df, report_df


def format_confusion_matrix(conf_matrix):
    index = [f"Actual {bool(i)}" for i in range(conf_matrix.shape[0])]
    columns = [f"Predicted {bool(i)}" for i in range(conf_matrix.shape[1])]
    return pd.DataFrame(conf_matrix, index=index, columns=columns)
