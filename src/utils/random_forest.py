from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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
                ("onehot", OneHotEncoder(handle_unknown="ignore"), ["speaker_affiliation", "speaker_name", "speaker_state"]),
                ("pca", pca_pipeline, "statement_embedding"),
            ],
            remainder="passthrough",
        )
        self.model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                (
                    "classifier",
                    # RandomForestClassifier(),
                    VotingClassifier(
                        estimators=[
                            ("ab", AdaBoostClassifier()),
                            ("gb", GradientBoostingClassifier()),
                        ],
                        voting="soft",
                    ),
                ),
            ]
        )

    def _format_data(self, df: pd.DataFrame):
        train_df, validate_df = preprocess_data(df)
        X_train = train_df.drop(
            columns=[
                "label",
                "statement",
                "subjects",
                # "speaker_name",
                "speaker_job",
                # "speaker_state",
                "statement_context",
                "prompt",
            ]
            + [c for c in train_df.columns if "subject" in c]
        )
        y_train = train_df["label"]

        X_test = validate_df.drop(columns=["label"])
        y_test = validate_df["label"]

        return X_train, X_test, y_train, y_test

    def train(self, df: pd.DataFrame):
        X_train, X_test, y_train, y_test = self._format_data(df)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        with open(os.path.join(self.config["training_arguments"]["output_dir"], "random_forest.pkl"), "wb") as f:
            pickle.dump(self.model, f)

        # Calculate metrics
        metrics_dict, conf_matrix_df, report_df = calculate_metrics(predictions=y_pred, references=y_test)

        display(pd.DataFrame.from_dict([metrics_dict]))
        display(report_df)
        display(conf_matrix_df)

        if hasattr(self.model.named_steps["classifier"], "feature_importances_"):
            importances = self.model.named_steps["classifier"].feature_importances_
            feature_names = X_train.columns
            feature_importances = sorted(zip(importances, feature_names), reverse=True)

            not_important = []
            for importance, name in feature_importances:
                print(f"{name}: {importance}")
                if importance < 1e-7:
                    not_important.append(name)

            print(sorted(not_important))
