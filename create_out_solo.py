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
    parser.add_argument('--data_root',default='/ubc/cs/home/g/gbhatt/borg/ranking/data/philipphager___baidu-ultr_uva-mlm-ctr/clicks/0.1.0/60cc071890b9bcc27adbfc78a642f1fa5d1668d90fadbe5b9fedcf3cd37bc89f/')
    parser.add_argument('--part', default=0, type=int)
    parser.add_argument('--out_dir', default='/ubc/cs/home/g/gbhatt/borg/ranking/data/custom_click', type=str)
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
    
    dataset = load_dataset(
                    "philipphager/baidu-ultr_uva-mlm-ctr",
                    name="annotations",
                    split="test",
                    cache_dir="~/.cache/huggingface",
                )

    click_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    dataset.save_to_disk(output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)