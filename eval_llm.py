
import argparse
from pathlib import Path
import os
import pdb
import glob
import logging
import torch
import json
import tqdm
import re
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from engine import local_trainer, Evaluator
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from datasets import load_dataset, load_from_disk, concatenate_datasets
from src.data import collate_fn, collate_fn_llm
from src.utils import get_rank, eval_ultr, get_ndcg, binary_accuracy, eval_ultr_ideal, eval_llm


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_gpus', default=4, type=int,
                        help="Number of GPUs available")
    
    parser.add_argument('--save_fname', default=None, type=str)

    # dataset parameters
    parser.add_argument('--output_path', default='/home/ec2-user/workspace/cf_ranking/outputs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data_path', default='/home/ec2-user/workspace/data/custom_click/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_folder', default='demo',
                        help='path where to save, empty for no saving')
    parser.add_argument('--load_path', default='',
                        help='path where to load model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--eval_llm', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--llm_exp', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_online', action='store_true', help='generate the data for online LLM evaluation')
    parser.add_argument('--eval_offline', action='store_true', help='offline evvaluation from the inference of the LLM')

    return parser


def parse_inp(input_path, filename):
    
    prob_patterns = [
        r"P\(purchase\)\s*=\s*([\d.]+)",
        r"Final Probability[:\s*\*]*([\d.]+)"
    ]
    item_patterns = [
        r"Item to be purchased\s*=\s*(.+)",
        r"Recommended item\s*[-:>\s]*(.+)"
    ]
    
    combined_rows = []

    with open(input_path, "r") as infile:
        for line_num, line in enumerate(infile):
            record = json.loads(line)

            # Extract user input
            model_input = record.get("modelInput", {})
            messages = model_input.get("messages", [])
            input_text = ""
            query = None
            products = None

            for msg in messages:
                if msg.get("role") == "user":
                    for content_piece in msg.get("content", []):
                        if content_piece.get("type") == "text":
                            content = content_piece["text"]
                            input_text += content

                            # Extract query and products JSON
                            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', content)
                            query_id_match = re.search(r'"query_id"\s*:\s*"([^"]+)"', content)
                            if query_match and query_id_match:
                                query = query_match.group(1)
                                query_id = query_id_match.group(1)

                            products_match = re.search(r'"products"\s*:\s*({.*?})\s*}', content, re.DOTALL)
                            if products_match:
                                products_json_str = products_match.group(1) + "}"
                                products = json.loads(products_json_str)
            # Extract model output
            model_output = record.get("modelOutput", {})
            output_parts = model_output.get("content", [])
            output_text = ""
            for part in output_parts:
                if part.get("type") == "text":
                    output_text += part["text"]

            # Extract fields
            probability = None
            for pattern in prob_patterns:
                match = re.search(pattern, output_text)
                if match:
                    try:
                        probability = float(match.group(1).strip())
                        break
                    except ValueError:
                        continue

            item_selected = None
            for pattern in item_patterns:
                match = re.search(pattern, output_text)
                if match:
                    raw_item = match.group(1).strip()
                    id_match = re.search(r'\b([A-Z0-9]{8,15})\b', raw_item)
                    if id_match:
                        item_selected = id_match.group(1)
                        break

            try:
                assert query
                assert products
                assert probability
                assert item_selected
                
                product_keys = list(products.keys())
                item_position = product_keys.index(item_selected)
                
                combined_rows.append({
                    "source_file": filename,
                    "query": query,
                    "products": json.dumps(products),
                    "purchase_prob": probability,
                    "item_selected": item_selected,
                    "query_id":query_id,
                    "item_position": item_position,
                    "output":output_text,
                    "input":input_text,
                })
            except:
                print(f"[{filename} Line {line_num}] Skipped: missing values")
    print (f'Processed files:{len(combined_rows)} skipped :{line_num - len(combined_rows)}')
    return pd.DataFrame(combined_rows)

def reorder_products(default_order, new_order, products):
    products = json.loads(products)
    product_ids = list(products.keys())
    assert len(product_ids) == len(default_order), "Mismatch in default_order and products"
    reordered = {product_ids[i]: products[product_ids[i]] for i in new_order}
    return reordered

def run_ranker(args):
    train_files = glob.glob(os.path.join(args.data_path, '*'))
    datasets = [load_from_disk(d) for d in train_files]
    dataset = concatenate_datasets(datasets)
    
    dataset = dataset.filter(lambda example: example['query_id'] != -1)

    with open(args.data_path.replace('processed', 'split_indices.json'), "r") as f:
        data_ids = json.load(f)

    test_dataset = dataset.filter(lambda example: example['query_id'] in data_ids['test'])
    collate_fn = collate_fn_llm
    
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                 num_workers=args.num_workers, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = local_trainer(train_loader=test_dataloader,val_loader=test_dataloader,
                            test_dataset=test_dataset,args=args)
    trainer.to(device)
    
    trainer.resume(load_path=args.load_path, model='arranger')
    
    reordered_batch = []
    trainer.eval()
    
    with torch.no_grad():        
        for idx, batch in tqdm.tqdm(enumerate(test_dataloader)):
            feat = torch.tensor(batch['query_document_embedding'], dtype=torch.float).to(trainer.device)
            out_dict = trainer.arranger(inputs_embeds=feat)
            
            new_positions = eval_llm(batch=batch, pred_scores=out_dict['logits'],
                                    device=trainer.device, args=args)
            #new_pos.append(new_positions.detach().cpu().tolist())
            temp_reorder = []
            for i in range(len(batch['query_id'])):
                
                reorder = reorder_products(
                        batch["position"][i],
                        new_positions[i],
                        batch['products'][i]
                        )
                #temp_reorder.append(reorder)
                temp = {'query_id':batch['query_id'][i],
                        'query':batch['query'][i],
                        'products':reorder,
                        }
                reordered_batch.append(temp)
    
    output_path = f"{args.output_dir}/{args.output_folder}.jsonl.out"

    with open(output_path, "w") as f:
        for line in reordered_batch:
            entry = {"query_group_input": line}
            f.write(json.dumps(entry) + "\n")

    print(f"JSONL saved to: {output_path}")
        

def plot(df, outfile):

    df = test_df.copy()

    # Preprocessing
    df['products_dict'] = df['products'].apply(json.loads)
    df['num_candidates'] = df['products_dict'].apply(len)
    df['selected_in_candidates'] = df.apply(
        lambda row: row['item_selected'] in row['products_dict'], axis=1
    )

    def get_selected_position(row):
        try:
            product_ids = list(row['products_dict'].keys())
            if row['item_selected'] in product_ids:
                return product_ids.index(row['item_selected']) + 1
            else:
                return -1
        except:
            return -1

    df['selected_position'] = df.apply(get_selected_position, axis=1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    # Plot 1: Purchase Probability
    sns.histplot(df['purchase_prob'], bins=20, kde=True, ax=axes[0])
    axes[0].set_title("Purchase Probability")
    axes[0].set_xlabel("Probability")
    axes[0].set_ylabel("Frequency")

    # Plot 2: Selected Item Position
    sns.countplot(
        x='selected_position',
        data=df,
        order=sorted(df['selected_position'].unique()),
        stat="probability",
        ax=axes[1]
    )
    axes[1].set_title("Selected Item Position")
    axes[1].set_xlabel("Position (-1 = not found)")
    axes[1].set_ylabel("Relative Frequency")

    # Finalize
    plt.tight_layout()
    plt.savefig(outfile)
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rank BERT', parents=[get_args_parser()])
    args = parser.parse_args()
    args.output_dir = args.output_path+args.output_folder
    args.log_file = open(args.output_dir+'/out_eval.log','a')
    
    if args.eval_online:
        run_ranker(args)
    else:
        inp_file = os.path.join(args.output_dir, 
                                f"batch_inference_purchase-labeln-{args.output_folder.split('_')[-1]}_0.jsonl.out")
        
        out  = args.output_folder.replace('_','-')
        inp_file = os.path.join(args.output_dir, 
                                 f"batch_inference_t-{out}_0.jsonl.out")
        
        llm_df = parse_inp(inp_file, inp_file)

        train_files = glob.glob(os.path.join(args.data_path, '*'))
        datasets = [load_from_disk(d) for d in train_files]
        dataset = concatenate_datasets(datasets)
        
        dataset = dataset.filter(lambda example: example['query_id'] != -1)

        with open(args.data_path.replace('processed', 'split_indices.json'), "r") as f:
            data_ids = json.load(f)

        test_dataset = dataset.filter(lambda example: example['query_id'] in data_ids['test'])
        
        test_df = test_dataset.to_pandas()

        test_max = test_df.loc[test_df.groupby('query_id')['purchase_prob'].idxmax()]
        llm_df_max = llm_df.loc[llm_df.groupby('query_id')['purchase_prob'].idxmax()]
        #merged = llm_df_max.merge(test_max, on='query_id', suffixes=('_df', '_test'))

        llm_df_max['query_id'] = llm_df_max['query_id'].astype(int)
        test_max['query_id'] = test_max['query_id'].astype(int)

        print(f"\nBefore - Expected purchase prob: E[p(pur)] = {np.mean(test_dataset['purchase_prob']):.4f}")
        print(f"After  - Expected purchase prob: E[p(pur)] = {llm_df['purchase_prob'].mean():.4f}")
        
        def get_cutoff(threshold_high, threshold_low=0.0):
            test_filtered = test_max[test_max['purchase_prob'] < threshold_high]
            test_filtered = test_filtered[test_filtered['purchase_prob'] >= threshold_low]

            query_ids_below_thresh = test_filtered['query_id']
            df_filtered = llm_df_max[llm_df_max['query_id'].isin(query_ids_below_thresh)]

            merged = df_filtered.merge(test_filtered, on='query_id', suffixes=('_df', '_test'))
            merged['purchase_prob_diff'] = (merged['purchase_prob_df'] - merged['purchase_prob_test']).abs()

            mean_before = merged['purchase_prob_test'].mean()
            mean_after = merged['purchase_prob_df'].mean()

            se_before = merged['purchase_prob_test'].std(ddof=1) / np.sqrt(len(merged))
            se_after = merged['purchase_prob_df'].std(ddof=1) / np.sqrt(len(merged))

            print(f"\nCut-off {threshold_low}:{threshold_high}  Before - Expected purchase prob: "
                f"E[p(pur) in {threshold_low}:{threshold_high}] = {mean_before:.4f} ± {se_before:.4f}")
            print(f"Cut-off {threshold_low}:{threshold_high}  After  - Expected purchase prob: "
                f"E[p(pur) in {threshold_low}:{threshold_high}] = {mean_after:.4f} ± {se_after:.4f}\n")

        get_cutoff(1.0, 0.8)
        get_cutoff(0.8, 0.6)
        get_cutoff(0.6, 0.4)
        get_cutoff(0.4)
        #get_cutoff(0.2)

        plot(test_df, os.path.join(args.output_dir, 
                                f"testdf.jpg"))