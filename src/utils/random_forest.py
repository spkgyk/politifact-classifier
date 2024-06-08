from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from IPython.display import display
from typing import Dict
import pandas as pd
import numpy as np
import pickle
import os


from .get_metrics import calculate_metrics
from .preprocess import preprocess_data


# Custom transformer to handle ADA embeddings
class AdaEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([embedding[:256] for embedding in X])


class MyModel:
    def __init__(self, config: Dict):
        self.config = config
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("tfidf_statement", TfidfVectorizer(stop_words="english"), "statement"),
                ("tfidf_statement_context", TfidfVectorizer(stop_words="english"), "statement_context"),
                ("onehot", OneHotEncoder(handle_unknown="ignore"), ["speaker_name", "speaker_state", "speaker_affiliation"]),
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
                    #     estimators=[("rf", RandomForestClassifier()), ("gb", GradientBoostingClassifier())],
                    #     voting="soft",
                    # ),
                ),
            ]
        )

    def train(self, df: pd.DataFrame):
        df = preprocess_data(df)
        X = df.drop(columns=["Label", "subjects", "speaker_job"])
        y = df["label"]
        # X["ada_embedding"] = X["ada_embedding"].apply(eval)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        with open(os.path.join(self.config["training_arguments"]["output_dir"], "random_forest.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        # Calculate metrics
        metrics_dict, conf_matrix_df, report_df, accuracy_df = calculate_metrics(y_test, y_pred)

        display(metrics_dict)
        display(conf_matrix_df)
        display(report_df)
        display(accuracy_df)
