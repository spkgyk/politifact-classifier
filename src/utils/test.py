from transformers.pipelines.pt_utils import KeyDataset
from joblib import Parallel, delayed
from transformers import pipeline
from datasets import Dataset
from tqdm.auto import tqdm
from typing import Any, Dict
from math import ceil
import pandas as pd

from .create_prompt import create_prompt


class ClassificationTester:

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.classifier = pipeline(task="text-classification", model=config["inference_model"], device_map="cuda")

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
        return self.classifier(create_prompt(json))[0]

    def __call__(self, *args: Any, **kwds: Any) -> Dict:
        return self.test(*args, **kwds)
