from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
)
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from joblib import Parallel, delayed
from IPython.display import display
from typing import Dict
import pandas as pd
import evaluate
import os

# import torch.nn as nn

from .create_prompt import create_prompt


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
        # self.model.classifier = nn.Linear(self.model.classifier.in_features, config["num_labels"])
        # self.model.num_labels = config["num_labels"]
        # self.model.config.num_labels = config["num_labels"]
        # self.model.config.problem_type == "single_label_classification" if config["num_labels"] == 2 else "multi_label_classification"
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.training_arguments = TrainingArguments(**config["training_arguments"])

        acc = evaluate.load("accuracy", trust_remote_code=True, average="weighted")
        prec = evaluate.load("precision", trust_remote_code=True, average="weighted")
        rec = evaluate.load("recall", trust_remote_code=True, average="weighted")
        f1 = evaluate.load("f1", trust_remote_code=True, average="weighted")
        mcc = evaluate.load("matthews_correlation", trust_remote_code=True)
        self.metrics = evaluate.combine([acc, prec, rec, f1, mcc]) if config["num_labels"] == 2 else evaluate.combine([acc, prec, rec, f1])

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
        train_end = int(0.8 * len(df))

        # Split the DataFrame
        train_df = df.iloc[:train_end]
        validate_df = df.iloc[train_end:]

        # Convert to HF dataset
        dataset = DatasetDict()
        dataset["train"] = Dataset.from_pandas(train_df)
        dataset["validation"] = Dataset.from_pandas(validate_df)
        return dataset

    def tokenize(self, entry):
        return self.tokenizer(entry["prompt"])

    def compute_metrics(self, pred):
        references = pred.label_ids
        predictions = pred.predictions.argmax(-1)
        metrics = self.metrics.compute(predictions=predictions, references=references)
        return metrics

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
        metrics = pd.DataFrame([self.trainer.evaluate()])
        display(metrics)
        self.trainer.train()
        metrics = pd.DataFrame([self.trainer.evaluate()])
        display(metrics)
        output_path = os.path.join(self.config["training_arguments"]["output_dir"], self.config["model_name"])
        self.trainer.save_model(output_path)
