from datasets import load_dataset

dataset = load_dataset(
    "philipphager/baidu-ultr_baidu-mlm-ctr",
    name="clicks",
    split="train", # ["train", "test"]
    cache_dir="~/.cache/huggingface",
)

dataset = load_dataset(
    "philipphager/baidu-ultr_baidu-mlm-ctr",
    name="clicks",
    split="test", # ["train", "test"]
    cache_dir="~/.cache/huggingface",
)
