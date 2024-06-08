from transformers.pipelines.pt_utils import KeyDataset
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from transformers import pipeline
from datasets import Dataset
from typing import Any, Dict
from tqdm.auto import tqdm
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

    def test_df(self, df: pd.DataFrame):
        df["prompt"] = Parallel(n_jobs=-1)(delayed(create_prompt)(row) for _, row in df.iterrows())
        data = KeyDataset(Dataset.from_pandas(df), key="prompt")
        bs = self.config["inference_batch_size"]

        results = {}
        pbar = tqdm(enumerate(self.classifier(data, batch_size=bs)), total=len(data))
        for i, result in pbar:
            results[i] = result

        return results

    def test(self, json: Dict) -> Dict:
        output = self.classifier(create_prompt(json))[0]
        return output["label"] == "LABEL_1"

    def flip(self, json: Dict) -> str:
        truth_value = self.test(json)
        input = {"old_truth_value": truth_value, "new_truth_value": not truth_value, "statement": json["statement"]}
        response = self.ollama.invoke(LLAMA_TASK.format(**input))
        return response

    def ensemble_predict(self, json: Dict):
        # Get prediction from the sklearn model
        sklearn_pred_proba = self.random_forest.predict_proba(json)[0]  # assuming the model outputs probabilities

        # Get prediction from the BERT model
        bert_result = self.classifier(create_prompt(json))[0]
        bert_pred_proba = bert_result["score"]  # assuming the BERT model outputs a score (probability)

        # If BERT model has two classes, convert the score to a probability array
        if bert_result["label"] == "LABEL_0":
            bert_pred_proba = [bert_pred_proba, 1 - bert_pred_proba]
        else:
            bert_pred_proba = [1 - bert_pred_proba, bert_pred_proba]

        # Combine the probabilities (you can use averaging, weighted averaging, etc.)
        ensemble_proba = np.mean([sklearn_pred_proba, bert_pred_proba], axis=0)

        # Get the final prediction
        final_pred = np.argmax(ensemble_proba)

        return final_pred, ensemble_proba
