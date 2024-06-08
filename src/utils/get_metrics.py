from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import pandas as pd
import evaluate

acc = evaluate.load("accuracy", trust_remote_code=True, average="weighted")
prec = evaluate.load("precision", trust_remote_code=True, average="weighted")
rec = evaluate.load("recall", trust_remote_code=True, average="weighted")
f1 = evaluate.load("f1", trust_remote_code=True, average="weighted")
mcc = evaluate.load("matthews_correlation", trust_remote_code=True)
metrics = evaluate.combine([acc, prec, rec, f1, mcc])


def calculate_metrics(predictions, references):
    metrics_dict = metrics.compute(predictions=predictions, references=references)

    mcc = matthews_corrcoef(references, predictions)
    report = classification_report(references, predictions, output_dict=True)
    conf_matrix = confusion_matrix(references, predictions)

    # extract accuracy
    accuracy = report["accuracy"]
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop("accuracy")

    # print accuracy df
    accuracy_df = pd.DataFrame({"metric": ["accuracy", "Matthews Correlation Coefficient"], "value": [accuracy, mcc]})

    # reformat and print confusion matrix
    conf_matrix_df = pd.DataFrame(
        conf_matrix,
        index=[f"Actual {bool(i)}" for i in range(conf_matrix.shape[0])],
        columns=[f"Predicted {bool(i)}" for i in range(conf_matrix.shape[1])],
    )

    return metrics_dict, conf_matrix_df, report_df, accuracy_df
