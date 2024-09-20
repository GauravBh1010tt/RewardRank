import os
import tqdm
import torch
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from src.data import collate_fn
from datasets import load_dataset
from torch.utils.data import DataLoader
from bbm.src.data import collate_click_fn
from bbm.src.model import CrossEncoder, IPSCrossEncoder, PBMCrossEncoder
from collections import defaultdict

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default='train', choices=['train','val'])
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--st_part', default=0, type=int)
    parser.add_argument('--out_dir', default='/home/ggbhatt/workspace/data/custom_click_new/', type=str)
    parser.add_argument('--model', default='naive-pointwise', choices=['naive-pointwise', 
                                                                      'naive-listwise', 'pbm', 'dla', 
                                                                      'ips-pointwise', 'ips-listwise'])
    return parser


def main(args):

    # fix the seed for reproducibility
    file_name = 'all_'+args.split+str(args.part)+'.feather'
    output_file = os.path.join(args.out_dir,'temp',file_name)

    Path(os.path.join(args.out_dir, 'temp')).mkdir(parents=True, exist_ok=True)

    if args.split == 'train':
        big_max_parts = 27796 * args.st_part
        max_parts = 3475
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

    ultr_models = ['twotower', 'ips-pointwise']
    map_mod = {'twotower':['examination_2tower', 'relevance_2tower', PBMCrossEncoder],
               'ips-pointwise':['examination_ips', 'relevance_ips', IPSCrossEncoder]}
    models = {}
    for i in ultr_models:
        models[i]= map_mod[i][-1].from_pretrained("philipphager/baidu-ultr_uva-bert_"+i)

    batch_out = defaultdict(lambda: [])
    for idx, batch in tqdm.tqdm(enumerate(click_loader)):

        if idx<(big_max_parts + args.part*max_parts):
            continue
        if idx>(big_max_parts + (args.part+1)*max_parts):
            break

        row,col = batch['tokens'].shape[0], batch['tokens'].shape[1] 
        new_batch = {'tokens':batch['tokens'].reshape(row*col,128),
                'attention_mask':batch['attention_mask'].reshape(row*col,128),
                'token_types':batch['token_types'].reshape(row*col,128),
                'positions':batch['position'].reshape(row*col)}
        for i in ultr_models:
            out_c = models[i](new_batch)
            examination = np.array(out_c.examination).reshape(row,col)
            relevance = np.array(out_c.relevance).reshape(row,col)
            batch[map_mod[i][0]] = examination
            batch[map_mod[i][1]] = relevance

        for i,j in batch.items():
            batch_out[i].extend(j.tolist())
            #batch_out['click_mod'].extend(pred.tolist())

    df = pd.DataFrame(batch_out)
    df.to_feather(output_file)


def concat(args):
    print('\n concatenating temp files ...\n')
    files = glob.glob(os.path.join(args.out_dir, 'temp', '*'))

    assert len(files)==8

    out = pd.read_feather(files[0])

    for i in files[1:]:
        temp = pd.read_feather(i)
        out = pd.concat([out,temp])

    df = pd.DataFrame(out)
    file_name = 'all_'+args.split+str(args.st_part)+'.feather'
    output_file = os.path.join(args.out_dir,args.split,file_name)
    df.to_feather(output_file)
    
    print('\n concatenation complete ...\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.concat:
        concat(args)
    else:
        main(args)