import enum
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

import mmh3
import torch
import pdb
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split

from bbm.src.const import (
    SPECIAL_TOKENS,
    SEGMENT_TYPES,
    MAX_SEQUENCE_LENGTH,
)

COLUMNS = {
    "query_id": {"padded": False, "dtype": str},
    "query_document_embedding": {"padded": True, "dtype": float, "type": "bert"},
    "position": {"padded": True, "dtype": int},
    "mask": {"padded": True, "dtype": bool},
    "n": {"padded": False, "dtype": int},
    "click": {"padded": True, "dtype": int},
    "examination_ips": {"padded": True, "dtype": float},
    "relevance_ips": {"padded": True, "dtype": float},
    "examination_twotower": {"padded": True, "dtype": float},
    "relevance_twotower": {"padded": True, "dtype": float},
    "tokens": {"padded": True, "dtype": int},
    "attention_mask": {"attention_mask": True, "dtype": bool},
    "token_types": {"token_types": True, "dtype": int},
    "label": {"padded": True, "dtype": int},
    "frequency_bucket": {"padded": False, "dtype": int},
    "bm25": {"padded": True, "dtype": float, "type": "ltr"},
    "bm25_title": {"padded": True, "dtype": float, "type": "ltr"},
    "bm25_abstract": {"padded": True, "dtype": float, "type": "ltr"},
    "tf_idf": {"padded": True, "dtype": float, "type": "ltr"},
    "tf": {"padded": True, "dtype": float, "type": "ltr"},
    "idf": {"padded": True, "dtype": float, "type": "ltr"},
    "ql_jelinek_mercer_short": {"padded": True, "dtype": float, "type": "ltr"},
    "ql_jelinek_mercer_long": {"padded": True, "dtype": float, "type": "ltr"},
    "ql_dirichlet": {"padded": True, "dtype": float, "type": "ltr"},
    "document_length": {"padded": True, "dtype": int, "type": "ltr"},
    "title_length": {"padded": True, "dtype": int, "type": "ltr"},
    "abstract_length": {"padded": True, "dtype": int, "type": "ltr"},
}


class FeatureType(str, enum.Enum):
    BERT = "bert"
    LTR = "ltr"


@lru_cache(maxsize=None)
def filter_features(feature_type: FeatureType) -> List[str]:
    return [k for k, v in COLUMNS.items() if "type" in v and v["type"] == feature_type]


def collate_fn(samples: List[Dict[str, np.ndarray]]):
    """
    Collate function for training clicks / labels from the Baidu-ULTR-606k dataset:
    https://huggingface.co/datasets/philipphager/baidu-ultr-606k/blob/main/baidu-ultr-606k.py

    The function parses all available features, pads queries to the same numer of
    documents, and converts datatypes.
    """
    batch = defaultdict(lambda: [])
    max_n = int(max([sample["n"] for sample in samples]))

    #max_n = min(max_n, 50)

    #print(max_n)
    
    #pdb.set_trace()

    for sample in samples:
        
        all_tokens, all_token_types, all_attention_mask = [],[],[]
        for k in range(sample['n']):
            tokens, token_types, attention_mask = format_input(
                list(sample["query"]),
                list(sample["title"][k]),
                list(sample["abstract"][k]),
            )
            all_tokens.append(tokens)
            all_token_types.append(token_types)
            all_attention_mask.append(attention_mask)

        batch['tokens'].append(pad(all_tokens,max_n))
        batch['token_types'].append(pad(all_token_types,max_n))
        batch['attention_mask'].append(pad(all_attention_mask,max_n))

        for column, x in sample.items():
            if column in COLUMNS:
                #try:
                x = pad(x, max_n) if COLUMNS[column]["padded"] else x
                batch[column].append(x)
                
                # except:
                #     import pdb; pdb.set_trace()

        mask = pad(np.ones(sample["n"]), max_n).astype(bool)
        batch["mask"].append(mask)

    #import pdb; pdb.set_trace()
    return {
        column: np.array(features, dtype=COLUMNS[column]["dtype"])
        for column, features in batch.items()
    }


def format_input(
    query: List[int],
    title: List[int],
    abstract: List[int],
    max_tokens: int = MAX_SEQUENCE_LENGTH,
    special_tokens: Dict[str, int] = SPECIAL_TOKENS,
    segment_types: Dict[str, int] = SEGMENT_TYPES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Format BERT model input as:
    [CLS] + query + [SEP] + title + [SEP] + abstract + [SEP] + [PAD]
    """
    CLS = special_tokens["CLS"]
    SEP = special_tokens["SEP"]
    PAD = special_tokens["PAD"]

    query_tokens = [CLS] + query + [SEP]
    query_token_types = [segment_types["QUERY"]] * len(query_tokens)

    text_tokens = title + [SEP] + abstract + [SEP]
    text_token_types = [segment_types["TEXT"]] * len(text_tokens)

    tokens = query_tokens + text_tokens
    token_types = query_token_types + text_token_types

    padding = max(max_tokens - len(tokens), 0)
    tokens = tokens[:max_tokens] + padding * [PAD]
    token_types = token_types[:max_tokens] + padding * [segment_types["PAD"]]

    tokens = np.array(tokens, dtype=int)
    token_types = np.array(token_types, dtype=int)
    attention_mask = tokens > PAD

    return tokens, token_types, attention_mask


def collate_click_fn(
    batch: List[Dict],
    max_tokens: int = MAX_SEQUENCE_LENGTH,
    special_tokens: Dict[str, int] = SPECIAL_TOKENS,
    segment_types: Dict[str, int] = SEGMENT_TYPES,
) -> Dict:
    collated = defaultdict(lambda: [])
    for sample in batch:
        for k in range(sample["n"]):
            tokens, token_types, attention_mask = format_input(
                list(sample["query"]),
                list(sample["title"][k]),
                list(sample["abstract"][k]),
                max_tokens,
                special_tokens,
                segment_types,
            )

            collated["query_id"].append(sample["query_id"])
            collated["tokens"].append(tokens)
            collated["token_types"].append(np.asarray(token_types))
            collated["attention_mask"].append(attention_mask)
            collated["positions"].append(sample["position"][k])
            collated["click"].append(sample["click"][k])

    return {
        "query_id": np.asarray(collated["query_id"]),
        "tokens": np.stack(collated["tokens"], axis=0),
        "attention_mask": np.stack(collated["attention_mask"], axis=0),
        "token_types": np.stack(collated["token_types"], axis=0),
        "clicks": np.asarray(collated["click"]),
        "positions": np.asarray(collated["positions"]),
    }


def random_split(
    dataset: Dataset,
    shuffle: bool,
    random_state: int,
    test_size: float,
    stratify: Optional[str] = None,
):
    """
    Stratify a train/test split of a Huggingface dataset.
    While huggingface implements stratification, this function enables stratification
    on all columns, not only the dataset's class label.
    """
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        idx,
        stratify=dataset[stratify] if stratify else None,
        shuffle=shuffle,
        test_size=test_size,
        random_state=random_state,
    )
    return dataset.select(train_idx), dataset.select(test_idx)


def pad(x: np.ndarray, max_n: int):
    """
    Pads first (batch) dimension with zeros.

    E.g.: x = np.array([5, 4, 3]), n = 5
    -> np.array([5, 4, 3, 0, 0])

    E.g.: x = np.array([[5, 4, 3], [1, 2, 3]]), n = 4
    -> np.array([[5, 4, 3], [1, 2, 3], [0, 0, 0], [0, 0, 0]])
    """
    x = np.array(x)
    padding = max(max_n - x.shape[0], 0)
    pad_width = [(0, padding)]

    for i in range(x.ndim - 1):
        pad_width.append((0, 0))

    return np.pad(x, pad_width, mode="constant")


def hash_labels(x: np.ndarray, buckets: int, random_state: int = 0) -> np.ndarray:
    """
    Use a fast and robust non-cryptographic hash function to map class labels to
    a fixed number of buckets into the range(1, buckets + 1).
    E.g.: np.array([1301239102, 12039, 12309]) -> np.array([5, 1, 20])
    """

    def hash(i: int) -> int:
        hash_value = mmh3.hash(str(i), seed=random_state)
        bucket = hash_value % buckets
        return bucket + 1

    return np.array(list(map(hash, x)))


def discretize(x: np.ndarray, low: float, high: float, buckets: int):
    """
    Bucket a continuous variable into n buckets. Indexing starts at 1 to avoid
    confusion with the padding value 0.
    """
    boundaries = np.linspace(low, high, num=buckets + 1)
    return np.digitize(x, boundaries, right=False)


class LabelEncoder:
    def __init__(self):
        self.value2id = {}
        self.max_id = 1

    def __call__(self, x):
        if x not in self.value2id:
            self.value2id[x] = self.max_id
            self.max_id += 1

        return self.value2id[x]

    def __len__(self):
        return len(self.value2id)
