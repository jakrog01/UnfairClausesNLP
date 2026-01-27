import os

import requests
from datasets import load_dataset, Value


class UnfairClausesDataProvider:
    DATA_URLS = {
        "train": "https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl/resolve/main/train.csv",
        "test": "https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl/resolve/main/test.csv",
        "val": "https://huggingface.co/datasets/laugustyniak/abusive-clauses-pl/resolve/main/dev.csv",
    }

    @staticmethod
    def download_data(data_dir="data"):
        os.makedirs(data_dir, exist_ok=True)
        for split_name, url in UnfairClausesDataProvider.DATA_URLS.items():
            filepath = os.path.join(data_dir, f"{split_name}.csv")

            if not os.path.exists(filepath):
                response = requests.get(url)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    f.write(response.content)

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def _map_labels(self, example):
        example["label_name"] = example["label"]
        label_map = {"BEZPIECZNE_POSTANOWIENIE_UMOWNE": 0, "KLAUZULA_ABUZYWNA": 1}
        val = example.get("label")

        if isinstance(val, str):
            s = val.strip()
            if s.isdigit():
              example["label"] = int(s)
            else:
              example["label"] = label_map.get(s, -1)
        else:
            example["label"] = -1
        return example

    def load_data(self, split="train"):
        if split not in self.DATA_URLS:
            raise ValueError(f"Invalid split: {split}")

        filepath = os.path.join(self.data_dir, f"{split}.csv")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found for {split}")

        dataset = load_dataset("csv", data_files={split: filepath})[split]
        dataset = dataset.map(self._map_labels)
        dataset = dataset.cast_column("label", Value("int32"))

        return dataset
