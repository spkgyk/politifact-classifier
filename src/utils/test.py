from transformers.pipelines.pt_utils import KeyDataset
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from transformers import pipeline
from datasets import Dataset
from tqdm.auto import tqdm
from typing import Dict
import pandas as pd
import numpy as np
import pickle

from .create_prompt import create_prompt


LLAMA_TASK = "Flip the following statement from {old_truth_value} to {new_truth_value}: {statement}. Return only the flipped statement."
LLAMA_TASK = PromptTemplate(input_variables=["old_truth_value", "new_truth_value", "statement"], template=LLAMA_TASK)


class ClassificationTester:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.classifier = pipeline(task="text-classification", model=config["inference_model"], device_map="cuda")
        with open(config["inference_rf_model"], "rb") as f:
            self.random_forest: Pipeline = pickle.load(f)
        self.ollama = Ollama(model="llama3")

    def test_rf(self, json: Dict) -> bool:
        input_df = pd.DataFrame([json])
        if "statement_embedding" in input_df.columns and type(input_df["statement_embedding"]) != list:
            input_df["statement_embedding"] = eval(input_df["statement_embedding"])
        return self.random_forest.predict(input_df)[0].astype(bool)

    def test_df_rf(self, df: pd.DataFrame):
        if "statement_embedding" in df.columns and type(df["statement_embedding"].iloc[0]) != list:
            df["statement_embedding"] = Parallel(n_jobs=-1)(delayed(eval)(row["statement_embedding"]) for _, row in df.iterrows())
            df = pd.DataFrame(df)
        return self.random_forest.predict(df).astype(bool)

    def test_bert(self, json: Dict) -> bool:
        output = self.classifier(create_prompt(json))[0]
        return output["label"] == "LABEL_1"

    def test_df_bert(self, df: pd.DataFrame):
        df["prompt"] = Parallel(n_jobs=-1)(delayed(create_prompt)(row) for _, row in df.iterrows())
        data = KeyDataset(Dataset.from_pandas(df), key="prompt")
        bs = self.config["inference_batch_size"]

        results = []
        pbar = tqdm(self.classifier(data, batch_size=bs), total=len(data))
        for result in pbar:
            results.append(result["label"] == "LABEL_1")

        return results

    def test_ensemble(self, json: Dict):
        # Prepare the input data for the sklearn model
        input_df = pd.DataFrame([json])
        if "statement_embedding" in input_df.columns and type(input_df["statement_embedding"]) != list:
            input_df["statement_embedding"] = eval(input_df["statement_embedding"])
        sklearn_pred_proba = self.random_forest.predict_proba(input_df)[0]

        # Get prediction from the BERT model
        bert_result = self.classifier(create_prompt(json))[0]
        bert_pred_proba = bert_result["score"]

        # If BERT model has two classes, convert the score to a probability array
        if bert_result["label"] == "LABEL_0":
            bert_pred_proba = [bert_pred_proba, 1 - bert_pred_proba]
        else:
            bert_pred_proba = [1 - bert_pred_proba, bert_pred_proba]

        # Combine the probabilities (using averaging)
        ensemble_proba = np.mean([sklearn_pred_proba, bert_pred_proba], axis=0)

        # Get the final prediction
        final_pred = np.argmax(ensemble_proba)

        return bool(final_pred), ensemble_proba

    def test_df_ensemble(self, df: pd.DataFrame):
        df["prompt"] = Parallel(n_jobs=-1)(delayed(create_prompt)(row) for _, row in df.iterrows())
        data = KeyDataset(Dataset.from_pandas(df), key="prompt")

        if "statement_embedding" in df.columns and type(df["statement_embedding"].iloc[0]) != list:
            df["statement_embedding"] = Parallel(n_jobs=-1)(delayed(eval)(row["statement_embedding"]) for _, row in df.iterrows())
            df = pd.DataFrame(df)

        bs = self.config["inference_batch_size"]

        sklearn_pred_proba = self.random_forest.predict_proba(df)

        results = []
        pbar = tqdm(zip(self.classifier(data, batch_size=bs), sklearn_pred_proba), total=len(data))
        for bert_result, sk_result in pbar:
            bert_pred_proba = bert_result["score"]
            if bert_result["label"] == "LABEL_0":
                bert_pred_proba = [bert_pred_proba, 1 - bert_pred_proba]
            else:
                bert_pred_proba = [1 - bert_pred_proba, bert_pred_proba]

            ensemble_proba = np.mean([sk_result, bert_pred_proba], axis=0)
            final_pred = np.argmax(ensemble_proba)

            results.append(bool(final_pred))

        return results

    def flip(self, json: Dict) -> str:
        truth_value = self.test_bert(json)
        input = {"old_truth_value": truth_value, "new_truth_value": not truth_value, "statement": json["statement"]}
        response = self.ollama.invoke(LLAMA_TASK.format(**input))
        return response
