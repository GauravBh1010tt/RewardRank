import os
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.data import collate_fn
from datasets import load_dataset
from torch.utils.data import DataLoader
from bbm.src.data import collate_click_fn
from bbm.src.model import CrossEncoder
from collections import defaultdict

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default='train', choices=['train','val'])
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--out_dir', default='/home/ggbhatt/workspace/data/custom_click', type=str)
    parser.add_argument('--model', default='naive-pointwise', choices=['naive-pointwise', 
                                                                      'naive-listwise', 'pbm', 'dla', 
                                                                      'ips-pointwise', 'ips-listwise'])
    return parser


def main(args):

    # fix the seed for reproducibility
    file_name = 'all_'+args.split+str(args.part)+'.feather'
    output_file = os.path.join(args.out_dir,file_name)

    if args.split == 'train':
        max_parts = 27796
    else:
        max_parts = 9280

    dataset = load_dataset(
        "philipphager/baidu-ultr_uva-mlm-ctr",
        name="clicks",
        split=args.split,
        trust_remote_code=True,
    )

    click_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    ultr_models = ['naive-pointwise', 'twotower', 'dla', 
                                'ips-pointwise', 'ips-listwise']
    map_mod = {'naive-pointwise':'click_n_point', 'twotower':'click_2tower_point', 'dla':'click_dla_list', 
                                'ips-pointwise':'click_ips_point', 'ips-listwise':'click_ips_list'}
    models = {}
    for i in ultr_models:
        models[i]= CrossEncoder.from_pretrained("philipphager/baidu-ultr_uva-bert_"+i)

    batch_out = defaultdict(lambda: [])
    for idx, batch in tqdm.tqdm(enumerate(click_loader)):

        if idx<args.part*max_parts:
            continue
        if idx>(args.part+1)*max_parts:
            break

        row,col = batch['tokens'].shape[0], batch['tokens'].shape[1] 
        new_batch = {'tokens':batch['tokens'].reshape(row*col,128),
                'attention_mask':batch['attention_mask'].reshape(row*col,128),
                'token_types':batch['token_types'].reshape(row*col,128)}
        for i in ultr_models:
            out_c = models[i](new_batch)
            pred = np.array(out_c.click).reshape(row,col)
            batch[map_mod[i]] = pred

        for i,j in batch.items():
            batch_out[i].extend(j.tolist())
            #batch_out['click_mod'].extend(pred.tolist())

    df = pd.DataFrame(batch_out)
    df.to_feather(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)