from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed
from IPython.display import display
from typing import Tuple
import pandas as pd
import numpy as np

from utils.create_prompt import create_prompt

STATE_MAPPING = {
    "tex": "texas",
    "the united states": "united states",
    "washington state": "washington",
    "washington d.c.": "district of columbia",
    "washington dc": "district of columbia",
    "washington, d.c.": "district of columbia",
    "virgina": "virginia",
    "virginia director, coalition to stop gun violence": "virginia",
}


def standardize_state(state: str):
    return STATE_MAPPING.get(state, state) if pd.notnull(state) else state


def process_subjects(df: pd.DataFrame, print_out=False):
    unique_subjects = set(subject for subjects in df["subjects"] for subject in subjects.split("$"))

    if print_out:
        print("[")
        for subject in sorted(unique_subjects):
            column = df["subjects"].apply(lambda x: int(subject in x))
            print(f'"{subject}" ({sum(column)}),')
        print("]")

    subjects_data = {f"subject-{subject}": df["subjects"].apply(lambda x: int(subject in x)) for subject in sorted(unique_subjects)}

    subjects_df = pd.DataFrame(subjects_data)
    df = pd.concat([df, subjects_df], axis="columns")
    return df


def encode_categorical_data(df: pd.DataFrame):
    categorical_columns = ["speaker_name", "speaker_state", "speaker_affiliation"]
    encoder = OneHotEncoder(sparse_output=False)
    encoded_categorical = encoder.fit_transform(df[categorical_columns])

    # Select the 'subject-' columns
    subject_columns = [c for c in df.columns if "subject-" in c]
    subject_data = df[subject_columns].values

    # Combine encoded categorical data and subject data
    extra_data = np.hstack([encoded_categorical, subject_data])

    df["extra_data"] = list(extra_data)
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.map(lambda x: x.lower().replace('"', "").strip() if isinstance(x, str) else x)
    df["label"] = df["Label"].apply(lambda x: int(x.lower() in ["true", "mostly-true"]))
    df = df.drop(columns=["Label"])
    df["speaker_state"] = df["speaker_state"].map(standardize_state)
    df = process_subjects(df)
    df["statement_context"] = df["statement_context"].fillna("")

    df["prompt"] = Parallel(n_jobs=-1)(delayed(create_prompt)(row) for _, row in df.iterrows())

    if "statement_embedding" in df.columns:
        df["statement_embedding"] = Parallel(n_jobs=-1)(delayed(eval)(row["statement_embedding"]) for _, row in df.iterrows())

    df = encode_categorical_data(df)

    train_df, validate_df = train_test_split(df, test_size=0.2, random_state=540)

    # for c in train_df.columns:
    #     display(c)

    return train_df, validate_df
