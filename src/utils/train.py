from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from joblib import Parallel, delayed
from IPython.display import display
from typing import Dict
import pandas as pd
import evaluate
import os

# import torch.nn as nn

from .get_metrics import calculate_metrics
from .create_prompt import create_prompt
from .preprocess import preprocess_data


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

    def _format_data(self, df: pd.DataFrame):
        # get numerical values for class labels
        df = preprocess_data(df)
        df["prompt"] = Parallel(n_jobs=-1)(delayed(create_prompt)(row) for _, row in df.iterrows())

        df = pd.DataFrame(df[["prompt", "label"]])
        train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)

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
        self.metrics_dict, self.conf_matrix_df, self.report_df, self.accuracy_df = calculate_metrics(
            predictions=predictions, references=references
        )
        return self.metrics_dict

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
        # get untrained metrics
        self.trainer.evaluate()
        display(self.metrics_dict)
        display(self.conf_matrix_df)
        display(self.report_df)
        display(self.accuracy_df)

        self.trainer.train()

        # get trained metrics of best model
        self.trainer.evaluate()
        display(self.metrics_dict)
        display(self.conf_matrix_df)
        display(self.report_df)
        display(self.accuracy_df)
        output_path = os.path.join(self.config["training_arguments"]["output_dir"], self.config["model_name"])
        self.trainer.save_model(output_path)
