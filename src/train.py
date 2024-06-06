from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from langchain.prompts import PromptTemplate
from joblib import Parallel, delayed
from evaluate import load
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

    return PROMPT.format(**inputs).lower()


class ClassificationTrainer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=config["num_labels"],
            trust_remote_code=True,
            device_map="cuda",
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.training_arguments = TrainingArguments(**config["training_arguments"])

        self.accuracy_metric = load("accuracy", trust_remote_code=True)
        self.precision_metric = load("precision", trust_remote_code=True)
        self.recall_metric = load("recall", trust_remote_code=True)
        self.f1_metric = load("f1", trust_remote_code=True)
        if config["num_labels"] == 2:
            self.mcc_metric = load("matthews_correlation", trust_remote_code=True)

    def _define_labels(self, df: pd.DataFrame):
        if self.config["num_labels"] == 6:
            label_encoder = LabelEncoder()
            df["label"] = label_encoder.fit_transform(df["Label"])
        elif self.config["num_labels"] == 2:
            df["label"] = df["Label"].apply(lambda x: int("true" in x.lower()))

        return df

    def _format_data(self, df: pd.DataFrame):
        # get numerical values for class labels
        df = self._define_labels(df)

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
        return self.tokenizer(entry["prompt"])

    def compute_metrics(self, pred):
        references = pred.label_ids
        predictions = pred.predictions.argmax(-1)
        accuracy = self.accuracy_metric.compute(predictions=predictions, references=references)
        precision = self.precision_metric.compute(predictions=predictions, references=references, average="weighted")
        recall = self.recall_metric.compute(predictions=predictions, references=references, average="weighted")
        f1 = self.f1_metric.compute(predictions=predictions, references=references, average="weighted")
        if self.config["num_labels"] == 2:
            mcc = self.mcc_metric.compute(predictions=predictions, references=references)
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "mcc": mcc}
        else:
            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def train(self, df: pd.DataFrame):
        dataset = self._format_data(df)
        dataset = dataset.map(self.tokenize, batched=True)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            data_collator=self.data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        self.trainer.train()
        self.trainer.evaluate()
        self.trainer.save_model("true_false")
