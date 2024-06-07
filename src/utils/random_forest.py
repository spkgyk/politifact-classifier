from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from IPython.display import display
import pandas as pd
import numpy as np


# Custom transformer to split subjects
class SubjectSplitter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.unique_subjects = set()
        for subjects in X:
            self.unique_subjects.update(subjects.split("$"))
        return self

    def transform(self, X):
        subject_data = {subject: [] for subject in self.unique_subjects}
        for subjects in X:
            subject_set = set(subjects.split("$"))
            for subject in self.unique_subjects:
                subject_data[subject].append(int(subject in subject_set))
        return pd.DataFrame(subject_data)


# Custom transformer to handle ADA embeddings
class AdaEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([embedding[:256] for embedding in X])


class MyModel:
    def __init__(self):
        self.text_transformer = Pipeline(steps=[("tfidf", TfidfVectorizer(stop_words="english"))])
        self.subject_transformer = Pipeline(steps=[("splitter", SubjectSplitter())])
        self.categorical_features = ["speaker_name", "speaker_job", "speaker_state", "speaker_affiliation", "statement_context"]
        self.categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("text", self.text_transformer, "statement"),
                ("subject", self.subject_transformer, "subjects"),
                ("cat", self.categorical_transformer, self.categorical_features),
                # ("ada", AdaEmbeddingTransformer(), "ada_embedding"),
            ]
        )
        self.model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(),
                    # VotingClassifier(
                    # estimators=[("rf", RandomForestClassifier(400)), ("gb", GradientBoostingClassifier(n_estimators=400))], voting="soft"
                    # ),
                ),
            ]
        )

    def train(self, df):
        X = df.drop(columns=["Label"])
        y = df["Label"].apply(lambda x: int("true" in x.lower()))
        # X["ada_embedding"] = X["ada_embedding"].apply(eval)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        # Calculate metrics
        mcc = matthews_corrcoef(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # extract accuracy
        accuracy = report["accuracy"]
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.drop("accuracy")
        display(report_df)

        # print accuracy df
        accuracy_df = pd.DataFrame({"metric": ["accuracy", "Matthews Correlation Coefficient"], "value": [accuracy, mcc]})
        display(accuracy_df)

        # reformat and print confusion matrix
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=[f"Actual {bool(i)}" for i in range(conf_matrix.shape[0])],
            columns=[f"Predicted {bool(i)}" for i in range(conf_matrix.shape[1])],
        )
        display(conf_matrix_df)
