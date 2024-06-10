from transformers.pipelines.pt_utils import KeyDataset
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from transformers import pipeline
from typing import Dict, Iterator
from transformers import logging
from datasets import Dataset
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import warnings
import pickle
import os

from ..data import create_prompt

warnings.filterwarnings("ignore", message="You seem to be using the pipelines sequentially on GPU")
logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


LLAMA_TASK = """
Suppose that the following statement is "{old_truth_value}". Flip the statement to "{new_truth_value}".
Return ONLY the flipped statement, with no additional comments or formatting.

Examples:
Statement 1: Sarah Palin was repeating Abraham Lincoln's words in discussing the war in Iraq.
Flipped: Sarah Palin was not repeating Abraham Lincoln's words in discussing the war in Iraq.
Statement 2: Earmarked dollars have doubled just since 2000, and more than tripled in the last 10 years.
Flipped: Earmarked dollars have not doubled since 2000, and less than tripled in the last 10 years.

Statement: {statement}
Flipped:
"""
LLAMA_TASK = PromptTemplate(input_variables=["old_truth_value", "new_truth_value", "statement"], template=LLAMA_TASK)


class ClassificationTester:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.classifier = pipeline(task="text-classification", model=config["inference_model"], device_map="cuda")
        with open(config["inference_rf_model"], "rb") as f:
            self.random_forest: Pipeline = pickle.load(f)
        self.ollama = Ollama(model="llama3")

    def _evaluate_statement_embedding(self, df: pd.DataFrame) -> pd.DataFrame:
        if "statement_embedding" in df.columns and type(df["statement_embedding"].iloc[0]) != list:
            df["statement_embedding"] = Parallel(n_jobs=-1)(delayed(eval)(row["statement_embedding"]) for _, row in df.iterrows())
            df = pd.DataFrame(df)
        return df

    def test_rf(self, json: Dict) -> bool:
        input_df = pd.DataFrame([json])
        input_df = self._evaluate_statement_embedding(input_df)
        return self.random_forest.predict(input_df)[0].astype(bool)

    def test_df_rf(self, df: pd.DataFrame):
        df = self._evaluate_statement_embedding(df)
        return self.random_forest.predict(df).astype(bool)

    def test_bert(self, json: Dict) -> bool:
        output = self.classifier(create_prompt(json))[0]
        return output["label"] == "LABEL_1"

    def test_df_bert(self, df: pd.DataFrame):
        data = Dataset.from_pandas(df)
        data = data.map(lambda example: {"prompt": create_prompt(example)})
        data = KeyDataset(data, key="prompt")

        bs = self.config["inference_batch_size"]

        results = []
        for result in tqdm(self.classifier(data, batch_size=bs), total=len(data)):
            results.append(result["label"] == "LABEL_1")

        return results

    def _get_ensemble_proba(self, sklearn_pred_proba, bert_result):
        bert_pred_proba = bert_result["score"]
        if bert_result["label"] == "LABEL_0":
            bert_pred_proba = [bert_pred_proba, 1 - bert_pred_proba]
        else:
            bert_pred_proba = [1 - bert_pred_proba, bert_pred_proba]

        ensemble_proba = np.mean([sklearn_pred_proba, bert_pred_proba], axis=0)
        final_pred = bool(np.argmax(ensemble_proba))

        return final_pred

    def test_ensemble(self, json: Dict):
        input_df = pd.DataFrame([json])
        input_df = self._evaluate_statement_embedding(input_df)
        sklearn_pred_proba = self.random_forest.predict_proba(input_df)[0]
        bert_result = self.classifier(create_prompt(json))[0]
        final_pred = self._get_ensemble_proba(sklearn_pred_proba, bert_result)
        return final_pred

    def test_df_ensemble(self, df: pd.DataFrame):
        df = self._evaluate_statement_embedding(df)
        sklearn_pred_proba = self.random_forest.predict_proba(df)

        data = Dataset.from_pandas(df)
        data = data.map(lambda example: {"prompt": create_prompt(example)})
        data = KeyDataset(data, key="prompt")

        bs = self.config["inference_batch_size"]

        results = []
        for bert_result, sk_result in tqdm(zip(self.classifier(data, batch_size=bs), sklearn_pred_proba), total=len(data)):
            final_pred = self._get_ensemble_proba(sk_result, bert_result)
            results.append(final_pred)

        return results

    def flip(self, json: Dict) -> Iterator[str]:
        truth_value = self.test_bert(json)
        input = {"old_truth_value": truth_value, "new_truth_value": not truth_value, "statement": json["statement"]}
        response = self.ollama.stream(LLAMA_TASK.format(**input))
        for s in response:
            yield s
