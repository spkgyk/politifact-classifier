from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from langchain.prompts import PromptTemplate
from datasets import Dataset
import pandas as pd
import numpy as np


def create_prompt_template():
    template = """The speaker {speaker_name} ({speaker_info}) said: "{statement}" in the context of {statement_context}."""
    return PromptTemplate(input_variables=["speaker_name", "speaker_info", "statement", "statement_context"], template=template)


def get_speaker_info(speaker_job, speaker_state, speaker_affiliation):
    if speaker_job:
        speaker_info = f"{speaker_job}"
        if speaker_affiliation and speaker_state:
            speaker_info += f" from {speaker_state}, affiliated with {speaker_affiliation}"
        elif speaker_state:
            speaker_info += f" from {speaker_state}"
        elif speaker_affiliation:
            speaker_info += f", affiliated with {speaker_affiliation}"
    else:
        speaker_info = f"from {speaker_state}"
        if speaker_affiliation:
            speaker_info += f", affiliated with {speaker_affiliation}"
    return speaker_info


def generate_prompt(row, prompt_template):
    speaker_info = get_speaker_info(row["speaker_job"], row["speaker_state"], row["speaker_affiliation"])
    return prompt_template.format(
        speaker_name=row["speaker_name"],
        speaker_info=speaker_info,
        statement=row["statement"],
        statement_context=row["statement_context"],
    )


if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    shuffled_df = df.sample(frac=1, random_state=42)
    train_end = int(0.8 * len(df))
    validate_end = int(0.9 * len(df))

    train, validate, test = np.split(shuffled_df, [train_end, validate_end])
