from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from langchain.prompts import PromptTemplate
from datasets import Dataset
import pandas as pd
import numpy as np


def create_prompt(row):
    statement = row["statement"]
    speaker_name = row["speaker_name"]
    speaker_affiliation = row["speaker_affiliation"]
    speaker_job = row["speaker_job"]
    speaker_state = row["speaker_state"]
    statement_context = row["statement_context"]

    template_str = ""
    if pd.notna(speaker_name):
        template_str += f"the speaker {speaker_name.replace('-', ' ')} ("
    else:
        template_str += f"a speaker ("

    if pd.notna(speaker_affiliation):
        template_str += f"{speaker_affiliation.replace('-', ' ').lower()} "
    if pd.notna(speaker_job):
        template_str += f"{speaker_job} "
    if pd.notna(speaker_state):
        template_str += f"from {speaker_state}) "
    else:
        template_str += f") "

    template_str += f'said the statement: "{statement}"'

    if pd.notna(statement_context):
        template_str += f" in the context of {statement_context.lower()}."


if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Label"])

    shuffled_df = df.sample(frac=1, random_state=42)
    train_end = int(0.8 * len(df))
    validate_end = int(0.9 * len(df))

    train, validate, test = np.split(shuffled_df, [train_end, validate_end])
