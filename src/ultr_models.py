import os
import tqdm
import glob
import pdb
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
# from bbm.src.model import CrossEncoder, IPSCrossEncoder, PBMCrossEncoder
from collections import defaultdict

def infer_ultr(pos_idx, device, model='ips'):

    # fix the seed for reproducibility
    #print ('here')
    # if model == 'ips':
    #     model= IPSCrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_ips-pointwise")
    # else:
    #     model= PBMCrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_twotower")
    df = pd.read_csv("/home/ec2-user/workspace/cf_rank/bbm/propensities/global_all_pairs.csv")
    model = torch.zeros(50, dtype=torch.float64).to(device)
    positions = df["position"].values
    propensities = torch.tensor(df.iloc[:, 1].values).to(device)

    model[positions] = propensities
    examination = model[pos_idx]

    return examination



def infer_ultr1(examination, relevance, batch, pos_idx, device, model='ips'):

    # fix the seed for reproducibility
    #print ('here')
    # if model == 'ips':
    #     model= IPSCrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_ips-pointwise")
    # else:
    #     model= PBMCrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_twotower")

    pos_org = torch.tensor(batch['position']).to(device)
    look_pos = (1-(pos_idx == pos_org).int()).sum(dim=1)
    idx_tuple = []

    #pdb.set_trace()

    new_batch = defaultdict(lambda: [])
    for idx, n in enumerate(batch['n']):

        if look_pos[idx]:
            for k in range(n):
                if pos_org[idx][k]!=pos_idx[idx][k]:
                    new_batch['tokens'].append(batch['tokens'][idx][k])
                    new_batch['attention_mask'].append(batch['attention_mask'][idx][k])
                    new_batch['token_types'].append(batch['token_types'][idx][k])
                    new_batch['positions'].append(pos_idx[idx][k].cpu().numpy())

                    idx_tuple.append((idx,k))

    for keys in new_batch:
        new_batch[keys] = np.stack(new_batch[keys])

    #pdb.set_trace()
        
    out_c = model(new_batch)
    #pdb.set_trace()

    for i,idx in enumerate(idx_tuple):
        examination[idx[0], idx[1]] = out_c.examination[i].item()
        relevance[idx[0], idx[1]] = out_c.relevance[i].item()

    return examination, relevance