import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import numpy as np
import pyarrow.feather as feather
from datasets import Dataset, load_from_disk, concatenate_datasets
import os
import pdb
import re
import argparse
import unicodedata, re
import numpy as np
import glob
import json
import math
from numbers import Real

def parse_args():
    """
    Parses command-line arguments for input folder, sampled data path, and model name.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Parse and embed query-product pairs.")
    parser.add_argument("--input_folder", type=str, default="../data/llm_data_irr", help="Folder containing raw_data and processed folders")
    parser.add_argument("--input_sampled_path", type=str, default="../data/esci-data/df_sampled_irr_sampling.parquet", help="Path to sampled query data")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2", help="SentenceTransformer model name")
    parser.add_argument('--binary', action='store_true')
    return parser.parse_args()

def alnum_only(s): 
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()

def parse_inp(input_path, filename):
    """
    Parses a JSONL output file to extract query, products, model predictions, and metadata.

    Args:
        input_path (str): Path to the input .jsonl.out file.

    Returns:
        pd.DataFrame: Parsed rows including query, products, prediction, and selected item info.
    """

    mapp = {}
    input_sampled = pd.read_parquet("/ubc/cs/home/g/gbhatt/borg/ranking/data/esci-data/df_sampled_irr_sampling.parquet")
    for i, j in zip(input_sampled['query'], input_sampled['query_id']):
        mapp[i] = j

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

            model_input = record.get("modelInput", {})
            messages = model_input.get("messages", [])
            input_text = ""
            query = None
            products = None

            # Extract query and products from user message
            for msg in messages:
                if msg.get("role") == "user":
                    for content_piece in msg.get("content", []):
                        if content_piece.get("type") == "text":
                            content = content_piece["text"]
                            input_text += content

                            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', content)
                            query_id_match = re.search(r'"query_id"\s*:\s*(\d+)', content)

                            if query_match:
                                query = query_match.group(1)
                            if query_id_match:
                                query_id = int(query_id_match.group(1))

                            products_match = re.search(r'"products"\s*:\s*({.*?})\s*}', content, re.DOTALL)
                            if products_match:
                                products_json_str = products_match.group(1) + "}"
                                products = json.loads(products_json_str)

            # Extract output text from model
            model_output = record.get("modelOutput", {})
            output_parts = model_output.get("content", [])
            output_text = ""
            for part in output_parts:
                if part.get("type") == "text":
                    output_text += part["text"]

            # Extract purchase probability
            probability = None
            for pattern in prob_patterns:
                match = re.search(pattern, output_text)
                if match:
                    try:
                        probability = float(match.group(1).strip())
                        break
                    except ValueError:
                        continue

            # Extract selected item
            item_selected = None
            for pattern in item_patterns:
                match = re.search(pattern, output_text)
                if match:
                    raw_item = match.group(1).strip()
                    id_match = re.search(r'\b([A-Z0-9]{8,15})\b', raw_item)
                    if id_match:
                        item_selected = id_match.group(1)
                        break

            # Store parsed row
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
                    #"query_id": query_id,
                    "query_id": mapp[query],
                    "item_position": item_position
                })
                # if line_num==567:
                #     pdb.set_trace()
            except:
                #pdb.set_trace()
                print(f"[{filename} Line {line_num}] Skipped: missing values")

    print(f'Processed files: {len(combined_rows)} skipped: {line_num - len(combined_rows)}')
    return pd.DataFrame(combined_rows)

def parse_inp_binary(input_path, filename):
    """
    Parses a JSONL output file to extract query, products, model predictions, and metadata.

    Args:
        input_path (str): Path to the input .jsonl.out file.

    Returns:
        pd.DataFrame: Parsed rows including query, products, prediction, and selected item info.
    """
    prob_patterns = [
        r"Output:\s*D\(purchase\)\s*=\s*(yes|no|[01](?:\.\d+)?)",
        r"\bD\(purchase\)\s*=\s*(yes|no|[01](?:\.\d+)?)",
        r"Final\s*Probability\s*[:\-\s*\)]*\s*([01](?:\.\d+)?)"
    ]

    item_patterns = [
        r"Item\s*to\s*be\s*purchased\s*=\s*[\"'“”‘’]?(?P<item>.+?)[\"'“”‘’]?(?:\s*$|\n)",
        r"Recommended\s*item\s*[-:>\s]*[\"'“”‘’]?(?P<item>.+?)[\"'“”‘’]?(?:\s*$|\n)"
    ]

    combined_rows = []

    log_file = open('/ubc/cs/home/g/gbhatt/borg/ranking/CF_ranking'+'/out.log','w')

    with open(input_path, "r") as infile:
        for line_num, line in enumerate(infile):
            record = json.loads(line)

            model_input = record.get("modelInput", {})
            messages = model_input.get("messages", [])
            input_text = ""
            query = None
            products = None

            # Extract query and products from user message
            for msg in messages:
                if msg.get("role") == "user":
                    for content_piece in msg.get("content", []):
                        if content_piece.get("type") == "text":
                            content = content_piece["text"]
                            input_text += content

                            query_match = re.search(r'"query"\s*:\s*"([^"]+)"', content)
                            query_id_match = re.search(r'"query_id"\s*:\s*(\d+)', content)

                            if query_match:
                                query = query_match.group(1)
                            if query_id_match:
                                query_id = int(query_id_match.group(1))

                            products_match = re.search(r'"products"\s*:\s*({.*?})\s*}', content, re.DOTALL)
                            if products_match:
                                products_json_str = products_match.group(1) + "}"
                                products = json.loads(products_json_str)

            # Extract output text from model
            model_output = record.get("modelOutput", {})
            output_parts = model_output.get("content", [])
            output_text = ""
            for part in output_parts:
                if part.get("type") == "text":
                    output_text += part["text"]

            # Extract purchase probability
            decision = None
            for pattern in prob_patterns:
                match = re.search(pattern, output_text)
                if match:
                    try:
                        decision = match.group(1).strip()
                        if decision in ['no', 'No', 'NO']:
                            decision = 0.0
                        else:
                            decision = 1.0
                        break
                    except ValueError:
                        continue

            # Extract selected item
            item_selected = None
            if decision:
                for pattern in item_patterns:
                    match = re.search(pattern, output_text)
                    if match:
                        raw_item = match.group(1).strip()
                        if raw_item:
                            id_match = re.findall(r'\b[A-Z0-9]{10}\b', raw_item)
                            if len(id_match)>0:
                                item_selected = id_match[0]
                            else:
                                for pid, pdata in products.items():
                                    #if alnum_only(raw_item.lower()) in alnum_only(pdata['product_title'].strip().lower()):
                                    if raw_item.lower() in pdata['product_title'].strip().lower():
                                        item_selected = pid
                                        break

            # # Store parsed row
            # try:
            #     assert query
            #     assert products
            #     # assert decision
            #     if decision:
            #         assert item_selected

            #     product_keys = list(products.keys())

            #     if item_selected:
            #         item_position = product_keys.index(item_selected)
            #     else:
            #         item_position = -1

            #     combined_rows.append({
            #         "source_file": filename,
            #         "query": query,
            #         "products": json.dumps(products),
            #         "purchase_prob": decision,
            #         "item_selected": item_selected,
            #         "query_id": query_id,
            #         "item_position": item_position
            #     })
            #     # if line_num==567:
            #     #     pdb.set_trace()
            # except:
            #     #pdb.set_trace()
            #     # if len(re.findall(r'\b[A-Z0-9]{9}\b', raw_item))>0:
            #     #     pdb.set_trace()
            #     # if line_num == 441 or line_num =='441':
            #     #     pdb.set_trace()
            #     print(f"[{filename} Line {line_num}] Skipped: missing values {raw_item}")
            #     print(f"[{filename} Line {line_num}] Skipped: missing values {raw_item}", file=log_file)

            # inside your outer loop...

            #pdb.set_trace()

            try:
                # --- required fields ---
                if not query:
                    raise ValueError("empty query")
                if not isinstance(products, dict) or not products:
                    raise ValueError("products must be a non-empty dict")

                # --- decision: exactly 0.0 or 1.0 (reject bool/NaN/others) ---
                if isinstance(decision, bool) or not isinstance(decision, Real):
                    raise ValueError(f"decision must be numeric 0.0 or 1.0, got {type(decision).__name__}={decision!r}")
                decision = float(decision)
                if math.isnan(decision) or decision not in (0.0, 1.0):
                    raise ValueError(f"decision must be exactly 0.0 or 1.0, got {decision!r}")

                # --- when decision==1.0, item_selected must exist and be in products ---
                if decision == 1.0:
                    if not item_selected:
                        raise ValueError("item_selected is required when decision == 1.0")
                    if item_selected not in products:
                        raise ValueError(f"item_selected {item_selected!r} not found in products")

                # compute position (only meaningful if item_selected provided and present)
                product_keys = list(products.keys())
                item_position = product_keys.index(item_selected) if item_selected in product_keys else -1

                combined_rows.append({
                    "source_file": filename,
                    "query": query,
                    "products": json.dumps(products, ensure_ascii=False),
                    "purchase_prob": decision,         # 0.0 or 1.0
                    "item_selected": item_selected,    # may be None/"" when decision==0.0
                    "query_id": query_id,
                    "item_position": item_position
                })

            except Exception as e:
                msg = f"[{filename} Line {line_num}] Skipped: {e}"
                print(msg)
                print(msg, file=log_file)

    print(f'Processed files: {len(combined_rows)} skipped: {line_num - len(combined_rows)}')
    return pd.DataFrame(combined_rows)

def gen_emb(df, output_path):
    """
    Generates query-product pair embeddings and saves them to disk.

    Args:
        df (pd.DataFrame): DataFrame containing query and product pairs.
        output_path (str): Path to save the HuggingFace dataset.
    """
    stacked_embeddings = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        query = row['query']
        products_json = row['products']
        products = json.loads(products_json)

        pair_texts = []
        for pid, pdata in products.items():
            title = pdata.get("product_title", "")
            brand = pdata.get("product_brand", "")
            color = pdata.get("product_color", "")
            pair_text = f"{query} [SEP] {title} Brand: {brand}. Color: {color}."
            pair_texts.append(pair_text)

        embeddings = model.encode(pair_texts, convert_to_tensor=True)
        stacked_embeddings.append(embeddings.cpu().numpy())

    df['query_item_embeddings'] = stacked_embeddings

    # Serialize embeddings as JSON strings
    df['query_item_embeddings'] = df['query_item_embeddings'].apply(
        lambda arr: json.dumps(arr.tolist()) if arr is not None else None
    )

    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.save_to_disk(output_path)

    print(f"Saved final DataFrame with embeddings to {output_path}")

def create_train_test_splits(input_folder, out_folder):

    train_files = glob.glob(os.path.join(out_folder, '*'))

    datasets = [load_from_disk(d) for d in train_files]
    full_dataset = concatenate_datasets(datasets)

    # Given length of DataFrame
    total_len = len(set(full_dataset['query']))

    # Generate and shuffle indices
    np.random.seed(42)  # for reproducibility
    all_indices = np.random.permutation(total_len)

    # Split into val, test, and train
    val_indices = all_indices[:4000].tolist()
    test_indices = all_indices[4000:8000].tolist()
    train_indices = all_indices[8000:].tolist()

    # Store in dict
    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices
    }

    # Save to JSON
    with open(os.path.join(input_folder, "split_indices.json"), "w") as f:
        json.dump(splits, f, indent=2)

    print("Done splitting ..\n")

def create_embeddings(args):
    """
    Main pipeline function that loads input, parses model outputs, and computes embeddings.
    """
    global model  # Retain scope to match existing variable use

    input_folder = args.input_folder
    model = SentenceTransformer(args.model_name)

    # Map query text to query ID
    # mapp = {}
    # input_sampled = pd.read_parquet(args.input_sampled_path)
    # for i, j in zip(input_sampled['query'], input_sampled['query_id']):
    #     mapp[i] = j

    # Process each file in raw_data
    #pdb.set_trace()
    
    for filename in os.listdir(os.path.join(input_folder, 'raw_data')):
        if not filename.endswith(".jsonl.out"):
            continue

        input_path = os.path.join(input_folder, 'raw_data', filename)
        out_id = int(filename.split('_')[-1].split('.')[0])

        #if out_id>4:
        out_folder = os.path.join(input_folder, 'processed', str(out_id))

        if args.binary:
            out_df = parse_inp_binary(input_path, filename=filename)
        else:
            out_df = parse_inp(input_path, filename=filename)

        gen_emb(out_df, out_folder)

    create_train_test_splits(input_folder, os.path.join(input_folder, 'processed'))

if __name__ == "__main__":
    args = parse_args()
    create_embeddings(args)
    #create_train_test_splits(args.input_folder, os.path.join(args.input_folder, 'processed'))