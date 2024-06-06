from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


# Custom transformer to split subjects
class SubjectSplitter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Extract unique subjects across all rows
        self.unique_subjects = set()
        for subjects in X:
            self.unique_subjects.update(subjects.split("$"))
        return self

    def transform(self, X):
        # Create a DataFrame with binary columns for each unique subject
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
        # Truncate each embedding to the first 256 elements
        return np.array([embedding[:256] for embedding in X])


# Load data
data = pd.read_csv("data/data_ada.csv")

# Features and Labels
X = data.drop(columns=["Label"])
y = data["Label"].apply(lambda x: int("true" in x.lower()))

# Convert ADA embeddings from string to numerical list
X["ada_embedding"] = X["ada_embedding"].apply(eval)

# Text Vectorization
# text_transformer = Pipeline(steps=[("tfidf", TfidfVectorizer(stop_words="english"))])

# Categorical Encoding for subjects
subject_transformer = Pipeline(steps=[("splitter", SubjectSplitter())])

# Categorical Encoding for other features
categorical_features = ["speaker_name", "speaker_job", "speaker_state", "speaker_affiliation", "statement_context"]
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # ("text", text_transformer, "statement"),
        ("subject", subject_transformer, "subjects"),
        ("cat", categorical_transformer, categorical_features),
        ("ada", AdaEmbeddingTransformer(), "ada_embedding"),
    ]
)

# Model Pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            VotingClassifier(estimators=[("rf", RandomForestClassifier()), ("gb", GradientBoostingClassifier())], voting="soft"),
        ),
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model
print("training...")
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(matthews_corrcoef(y_test, y_pred))
