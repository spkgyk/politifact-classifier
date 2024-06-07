from transformers.pipelines.pt_utils import KeyDataset
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from joblib import Parallel, delayed
from transformers import pipeline
from datasets import Dataset
from typing import Any, Dict
from tqdm.auto import tqdm
import pandas as pd

from .create_prompt import create_prompt


LLAMA_TASK = "Flip the following statement from {old_truth_value} to {new_truth_value}: {statement}. Return only the flipped statement."
LLAMA_TASK = PromptTemplate(input_variables=["old_truth_value", "new_truth_value", "statement"], template=LLAMA_TASK)


class ClassificationTester:

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.classifier = pipeline(task="text-classification", model=config["inference_model"], device_map="cuda")
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

    def __call__(self, *args: Any, **kwds: Any) -> Dict:
        return self.test(*args, **kwds)
