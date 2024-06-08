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

    # reformat and print confusion matrix
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f"Actual {bool(i)}" for i in range(conf_matrix.shape[0])],
        columns=[f"Predicted {bool(i)}" for i in range(conf_matrix.shape[1])],
    )

    return {"accuracy": acc, "matthews_correlation": mcc}, conf_matrix_df, report_df
