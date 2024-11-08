import os
import tqdm
import glob
import pdb
import torch
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
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--split', default='train', choices=['train','test','ann'])
    parser.add_argument('--data_root',default='/home/ec2-user/.cache/huggingface/philipphager___baidu-ultr_baidu-mlm-ctr/clicks/0.1.0/de47677224a1f47590a60a5ffca5ea84f1b105020620c07694cee02566ce4218/')
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--out_dir', default='/home/ec2-user/workspace/data/custom_click_new', type=str)
    parser.add_argument('--model', default='naive-pointwise', choices=['naive-pointwise', 
                                                                      'naive-listwise', 'pbm', 'dla', 
                                                                      'ips-pointwise', 'ips-listwise'])
    return parser


def main(args):

    # fix the seed for reproducibility
    #print ('here')
    args.st = args.part

    out_folder = args.split+str(args.st)
    
    Path(os.path.join(args.out_dir,args.split)).mkdir(parents=True, exist_ok=True)

    output_folder = os.path.join(args.out_dir,args.split,out_folder)

    if args.split =='train':
        st = args.st
        end = 54
        num_files = 7
    elif args.split=='test':
        st = 54+args.st
        end = 72
        num_files = 3

    all_files = glob.glob(os.path.join(args.data_root,'*.arrow'))

    dataset = load_dataset(path=args.data_root,
                      data_files=all_files[st:min(st+num_files,end)])

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

    examination = {'twotower':[], 'ips-pointwise':[]}
    relevance = {'twotower':[], 'ips-pointwise':[]}

    batch_out = defaultdict(lambda: [])
    for idx, batch in tqdm.tqdm(enumerate(click_loader)):

        row,col = batch['tokens'].shape[0], batch['tokens'].shape[1] 
        new_batch = {'tokens':batch['tokens'].reshape(row*col,128),
                'attention_mask':batch['attention_mask'].reshape(row*col,128),
                'token_types':batch['token_types'].reshape(row*col,128),
                'positions':batch['position'].reshape(row*col)}
        
        for i in ultr_models:
            out_c = models[i](new_batch)
            exam = np.array(out_c.examination).reshape(row,col).tolist()
            rel = np.array(out_c.relevance).reshape(row,col).tolist()
            temp_e, temp_r = [],[]
            for k,l in enumerate(batch['n']):
                temp_e.append(exam[k][:l])
                temp_r.append(rel[k][:l])

            examination[i].extend(temp_e)
            relevance[i].extend(temp_r)

        #pdb.set_trace()

    dataset = dataset.add_column(name='examination_twotower', column = examination['twotower'])
    dataset = dataset.add_column(name='examination_ips', column = examination['ips-pointwise'])
    dataset = dataset.add_column(name='relevance_twotower', column = relevance['twotower'])
    dataset = dataset.add_column(name='relevance_ips', column = relevance['ips-pointwise'])

    dataset.save_to_disk(output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)