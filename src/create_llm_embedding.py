import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import numpy as np
import pyarrow.feather as feather
from datasets import Dataset
import os
import pdb
import re
import argparse

def parse_args():
    """
    Parses command-line arguments for input folder, sampled data path, and model name.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Parse and embed query-product pairs.")
    parser.add_argument("--input_folder", type=str, default="../data/llm_data", help="Folder containing raw_data and processed folders")
    parser.add_argument("--input_sampled_path", type=str, default="../data/esci-data/shopping_queries_dataset/df_sampled_en.parquet", help="Path to sampled query data")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2", help="SentenceTransformer model name")
    return parser.parse_args()

def parse_inp(input_path):
    """
    Parses a JSONL output file to extract query, products, model predictions, and metadata.

    Args:
        input_path (str): Path to the input .jsonl.out file.

    Returns:
        pd.DataFrame: Parsed rows including query, products, prediction, and selected item info.
    """
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
                            if query_match:
                                query = query_match.group(1)

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
                    "query_id": mapp[query],
                    "item_position": item_position
                })
            except:
                print(f"[{filename} Line {line_num}] Skipped: missing values")

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

def create_embeddings():
    """
    Main pipeline function that loads input, parses model outputs, and computes embeddings.
    """
    global model, mapp, filename  # Retain scope to match existing variable use

    args = parse_args()
    input_sampled = pd.read_parquet(args.input_sampled_path)
    input_folder = args.input_folder
    model = SentenceTransformer(args.model_name)

    # Map query text to query ID
    mapp = {}
    for i, j in zip(input_sampled['query'], input_sampled['query_id']):
        mapp[i] = j

    # Process each file in raw_data
    for filename in os.listdir(os.path.join(input_folder, 'raw_data')):
        if not filename.endswith(".jsonl.out"):
            continue

        input_path = os.path.join(input_folder, 'raw_data', filename)
        out_id = int(filename.split('_')[-1].split('.')[0])
        out_folder = os.path.join(input_folder, 'processed', str(out_id))

        out_df = parse_inp(input_path)
        gen_emb(out_df, out_folder)

if __name__ == "__main__":
    create_embeddings()