import pandas as pd
import json
import numpy as np
import os
import re
import argparse
import unicodedata, re
import numpy as np
import glob
import json
import math
from numbers import Real
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    """
    Parses command-line arguments for input folder, sampled data path, and model name.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Parse and embed query-product pairs.")
    parser.add_argument("--input_folder", type=str, default="/data/llm_data", help="Folder containing raw_data and processed folders")
    return parser.parse_args()

def alnum_only(s): 
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()

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

def plot(df, out_dir):
    pur = [i for i in df['purchase_prob']]
    sns.histplot(pd.DataFrame(pur), bins=15, kde=True, stat='probability')
    plt.legend([], [], frameon=False)
    plt.xlabel(r'$\hat{y} = P(pur)$')
    plt.ylabel('Relative Frequency')

    # Save the plot before showing it
    plt.savefig(f"{out_dir}/histogram_purchase_probs.png", dpi=300, bbox_inches='tight')

    pos = [i+1 for i in df['item_position']]
    item_positions_series = pd.Series(pos)

    # Calculate the relative frequency of each unique item
    frequency = item_positions_series.value_counts(normalize=True)  # normalize=True gives relative frequencies

    # Convert the result into a DataFrame
    df = frequency.reset_index()
    df.columns = ['Item Position', 'Relative Frequency']

    # Plot the bar plot
    #plt.figure(figsize=(12, 6))
    sns.barplot(x='Item Position', y='Relative Frequency', data=df)

    # Add title and labels
    #plt.title('Relative Frequency of Item Positions', fontsize=16)
    plt.xlabel('Item Position')
    plt.ylabel('Relative Frequency')
    plt.savefig(f"{out_dir}/histogram_pos.png", dpi=300, bbox_inches='tight')


def create_embeddings():
    """
    Main pipeline function that loads input, parses model outputs, and computes embeddings.
    """
    args = parse_args()
    input_folder = args.input_folder
    dfs = []

    for filename in os.listdir(os.path.join(input_folder, 'raw_data')):
        if not filename.endswith(".jsonl.out"):
            continue

        input_path = os.path.join(input_folder, 'raw_data', filename)
        #out_id = int(filename.split('_')[-1].split('.')[0])
        out_df = parse_inp_binary(input_path, filename=filename)
        dfs.append(out_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    out_folder = os.path.join(input_folder, 'plots')
    os.makedirs(out_folder, exist_ok=True)
    plot(combined_df, out_dir=out_folder)

if __name__ == "__main__":
    create_embeddings()