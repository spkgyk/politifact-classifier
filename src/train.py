from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from langchain.prompts import PromptTemplate
from datasets import Dataset, DatasetDict
from joblib import Parallel, delayed
from typing import Dict
import pandas as pd

TEMPLATE = """{speaker_name} ({speaker_affiliation}{speaker_job}{speaker_state}) said the statement: "{statement}"{statement_context}"""
PROMPT = PromptTemplate(
    input_variables=[
        "speaker_name",
        "speaker_affiliation",
        "speaker_job",
        "speaker_state",
        "statement",
        "statement_context",
    ],
    template=TEMPLATE,
)


def create_prompt(row):
    speaker_name = f"""the speaker {row["speaker_name"].replace("-", " ")}""" if pd.notna(row["speaker_name"]) else "a speaker"
    speaker_affiliation = f"{row['speaker_affiliation'].replace('-', ' ')} " if pd.notna(row["speaker_affiliation"]) else ""
    speaker_job = f"{row['speaker_job']} " if pd.notna(row["speaker_job"]) else ""
    speaker_state = f"from {row['speaker_state']} state" if pd.notna(row["speaker_state"]) else ""
    statement_context = f" in the context of {row['statement_context']}." if pd.notna(row["statement_context"]) else ""

    inputs = {
        "speaker_name": speaker_name,
        "speaker_affiliation": speaker_affiliation,
        "speaker_job": speaker_job,
        "speaker_state": speaker_state,
        "statement": row["statement"],
        "statement_context": statement_context,
    }

    return PROMPT.format(**inputs)


class ClassificationTrainer:
    def __init__(self, config: Dict) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=config["num_labels"])

    @staticmethod
    def _format_data(df: pd.DataFrame):
        # get numerical values for class lables
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["Label"])

        # create prompt from all data labels
        df["prompt"] = Parallel(n_jobs=-1)(delayed(create_prompt)(row) for _, row in df.iterrows())

        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = df.drop(
            columns=[
                "Label",
                "statement",
                "subjects",
                "speaker_name",
                "speaker_job",
                "speaker_state",
                "speaker_affiliation",
                "statement_context",
            ]
        )

        # Define the split indices
        train_end = int(0.85 * len(df))
        validate_end = int(0.9 * len(df))

        # Split the DataFrame
        train_df = df.iloc[:train_end]
        validate_df = df.iloc[train_end:validate_end]
        test_df = df.iloc[validate_end:]

        # Convert to HF dataset
        dataset = DatasetDict()
        dataset["train"] = Dataset.from_pandas(train_df)
        dataset["validation"] = Dataset.from_pandas(validate_df)
        dataset["test"] = Dataset.from_pandas(test_df)
        return dataset

    def tokenize(self, entry):
        return self.tokenizer(entry["text"], padding=True, truncation=True)

    def train(self, df: pd.DataFrame):
        dataset = self._format_data(df)
        print(dataset)


if __name__ == "__main__":
    # load data
    df = pd.read_csv("data/data.csv")
    trainer = ClassificationTrainer()
    trainer.train(df)
