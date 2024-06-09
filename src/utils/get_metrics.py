from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, accuracy_score
import pandas as pd


def calculate_metrics(predictions, references):

    report = classification_report(references, predictions, output_dict=True)
    conf_matrix = confusion_matrix(references, predictions)
    mcc = matthews_corrcoef(references, predictions)
    acc = accuracy_score(references, predictions)

    # extract accuracy
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop("accuracy")

    macro_avg = report_df.loc["macro avg"]
    precision = macro_avg["precision"]
    recall = macro_avg["recall"]
    f1_score = macro_avg["f1-score"]

    # reformat and print confusion matrix
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f"Actual {bool(i)}" for i in range(conf_matrix.shape[0])],
        columns=[f"Predicted {bool(i)}" for i in range(conf_matrix.shape[1])],
    )

    training_metrics = {"accuracy": acc, "matthews_correlation": mcc, "precision": precision, "recall": recall, "f1-score": f1_score}

    return training_metrics, conf_matrix_df, report_df
