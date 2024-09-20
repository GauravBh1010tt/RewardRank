from datasets import load_dataset
from torch.utils.data import DataLoader
from bbm.src.model import CrossEncoder, IPSCrossEncoder, PBMCrossEncoder
from src.data import collate_fn
import numpy as np
import torch
import pandas as pd
from collections import defaultdict
import tqdm


batch_size = 8
dataset = load_dataset(
    "philipphager/baidu-ultr_uva-mlm-ctr",
    name="clicks",
    split="test",
    trust_remote_code=True,
)

click_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=0,
)

# click_loader1 = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     collate_fn=collate_click_fn,
# )

#model = CrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_naive-pointwise")
model1 = IPSCrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_ips-pointwise")
#model2 = CrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_ips-listwise")
model4 = PBMCrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_twotower")