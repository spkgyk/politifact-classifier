from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    AutoTokenizer,
    Trainer,
)
from datasets import Dataset, DatasetDict
from IPython.display import display
from typing import Dict
import pandas as pd
import os

from .get_metrics import calculate_metrics
from .preprocess import preprocess_data
from .model import CustomModel


class ClassificationTrainer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
        self.model = CustomModel(
            config["model_name"],
            num_extra_dims=2967,
            num_labels=config["num_labels"],
        )
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.training_arguments = TrainingArguments(**config["training_arguments"])

    def _format_data(self, df: pd.DataFrame) -> DatasetDict:
        train_df, validate_df = preprocess_data(df)
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(train_df[["prompt", "label", "extra_data"]]),
                "validation": Dataset.from_pandas(validate_df[["prompt", "label", "extra_data"]]),
            }
        )
        dataset = dataset.map(self._tokenize, batched=True)
        return dataset

    def _tokenize(self, entry: Dict) -> Dict:
        tokenized_inputs = self.tokenizer(entry["prompt"])
        tokenized_inputs["extra_data"] = entry["extra_data"]
        return tokenized_inputs

    def _compute_metrics(self, pred) -> Dict:
        references = pred.label_ids
        predictions = pred.predictions.argmax(-1)
        metrics_dict, conf_matrix_df, report_df = calculate_metrics(predictions, references)
        self.metrics_dict = metrics_dict
        self.conf_matrix_df = conf_matrix_df
        self.report_df = report_df
        return metrics_dict

    def train(self, df: pd.DataFrame) -> None:
        dataset = self._format_data(df)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            data_collator=self.data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(3)],
        )

        self._evaluate_and_display()

        self.trainer.train()

        self._evaluate_and_display()
        self._save_model()

    def _evaluate_and_display(self) -> None:
        self.trainer.evaluate()
        display(pd.DataFrame.from_dict([self.metrics_dict]))
        display(self.report_df)
        display(self.conf_matrix_df)

    def _save_model(self) -> None:
        output_path = os.path.join(self.config["training_arguments"]["output_dir"], self.config["model_name"])
        self.trainer.save_model(output_path)
