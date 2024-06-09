from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from IPython.display import display
from typing import Dict
import pandas as pd
import numpy as np
import pickle
import os


from .get_metrics import calculate_metrics
from .preprocess import preprocess_data


def extract_embeddings(X):
    return np.vstack(X)


class MyModel:
    def __init__(self, config: Dict):
        self.config = config

        pca_pipeline = Pipeline(
            steps=[
                ("extract", FunctionTransformer(extract_embeddings, validate=False)),
                ("pca", PCA(n_components=64)),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                # ("tfidf_statement", TfidfVectorizer(stop_words="english"), "statement"),
                # ("tfidf_statement_context", TfidfVectorizer(stop_words="english"), "statement_context"),
                ("onehot", OneHotEncoder(handle_unknown="ignore"), ["speaker_affiliation"]),
                ("pca", pca_pipeline, "statement_embedding"),
            ],
            remainder="passthrough",
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

    def _format_data(self, df: pd.DataFrame):
        df = preprocess_data(df)
        data_x = df.drop(
            columns=[
                "label",
                "statement",
                "subjects",
                "speaker_name",
                "speaker_job",
                "speaker_state",
                # "speaker_affiliation",
                "statement_context",
            ]
            + [c for c in df.columns if "subject" in c]
        )
        data_y = df["label"]
        return train_test_split(data_x, data_y, test_size=0.2, random_state=42)

    def train(self, df: pd.DataFrame):
        X_train, X_test, y_train, y_test = self._format_data(df)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        with open(os.path.join(self.config["training_arguments"]["output_dir"], "random_forest.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        # Calculate metrics
        metrics_dict, conf_matrix_df, report_df = calculate_metrics(y_test, y_pred)

        display(pd.DataFrame.from_dict([metrics_dict]))
        display(report_df)
        display(conf_matrix_df)

        importances = self.model.named_steps["classifier"].feature_importances_
        feature_names = X_train.columns
        feature_importances = sorted(zip(importances, feature_names), reverse=True)

        not_important = []
        for importance, name in feature_importances:
            if importance < 1e-7:
                not_important.append(name)
            else:
                print(f"{name}: {importance}")

        print(sorted(not_important))
