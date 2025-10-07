import argparse
import time
import torch
import re
import pdb
import pandas as pd
import json
import pprint
import os
import random
import numpy as np
import fasttext
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

def seed_for_query(query_id: int, base_seed: int):
    return (base_seed * 1_000_003 + int(query_id)) & 0xFFFFFFFF

def load_model_and_tokenizer(model_id):
    """Loads a tokenizer and causal language model with specific configuration.

    Args:
        model_id (str): Path or identifier for the model.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).eval()
    model = torch.compile(model)
    return model, tokenizer

def extract_info_batch(text):
    """Extracts probability and item ID from a generated text.

    Args:
        text (str): The generated output text from the model.

    Returns:
        tuple: Extracted probability (float or None) and item ID (str or None).
    """
    prob_patterns = [
        r"P\(purchase\)\s*=\s*([\d.]+)",
        r"Final probability of purchase\s*[:\-]?\s*([\d.]+)",
        r"Final Probability\s*[:\-]?\s*([\d.]+)"
    ]
    item_patterns = [
        r"Item to be purchased\s*[:=\-]?\s*(\w+)",
        r"Recommended item\s*[:=\-]?\s*(\w+)"
    ]

    probabilities, items = [], []
    try:
        prob = next((float(m.group(1)) for p in prob_patterns if (m := re.search(p, text, re.IGNORECASE))), None)
        item = next((m.group(1).strip() for p in item_patterns if (m := re.search(p, text, re.IGNORECASE))), None)
    except:
        prob = None
        item = None

    return prob, item

def get_query_groups(df, locale="us", random_sampling=True, num_items=8, random_state=42):
    """Groups query and product data from a Parquet file.

    Args:
        df (pd.dataframe):  Pandas dataframe.
        locale (str): Locale filter (currently unused).

    Returns:
        dict: A dictionary mapping query_id to its corresponding query and product metadata.
    """
    result = {}

    for query_id, group in df.groupby("query_id"):
        query_text = group["query"].iloc[0]

        if random_sampling:
            k = min(num_items, len(group))
            selected = group.sample(n=k, random_state=random_state, replace=False)

            rng = random.Random(seed_for_query(query_id, random_state))
            idx_pos = list(range(len(selected)))
            rng.shuffle(idx_pos)
            selected = selected.iloc[idx_pos].reset_index(drop=True)
        else:        
            # Split products by label
            relevant = group[group["esci_label"] == "I"]
            others = group[group["esci_label"] != "I"]

            # Ensure deterministic or reproducible random sample if needed
            sample_size = max(0, num_items - len(relevant))
            others_sampled = others.sample(n=sample_size, random_state=random_state) if sample_size > 0 else others.iloc[0:0]

            # Concatenate and trim to 8
            selected = pd.concat([relevant, others_sampled]).iloc[:num_items]

        # Construct product dictionary without esci_label
        products = {}
        esci = {}

        for _, row in selected.iterrows():
            product_id = row["product_id"]
            products[product_id] = {
                #"esci_label": row["esci_label"],
                "product_title": row["product_title"],
                "product_brand": row["product_brand"],
                "product_color": row["product_color"]
            }

            esci[product_id] = row["esci_label"]

        result[query_id] = {"query": query_text, "query_id":query_id, "products": products, "esci_labels": esci}
    return result

def generate_predictions(model, tokenizer, query_groups, batch_size, max_tokens, temperature, debug, args):
    """Generates purchase probability predictions using a language model.

    Args:
        model (torch.nn.Module): The causal language model.
        tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
        query_groups (dict): Mapping of query_id to query and product metadata.
        batch_size (int): Number of examples per batch.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        debug (bool): Whether to enable debug mode.
        args (Namespace): Parsed command-line arguments.

    Returns:
        pd.DataFrame: DataFrame of prediction results with query, product, and selected item.
    """
    all_records = []
    keys = list(query_groups.keys())[args.file_idx*args.chunk_size:(args.file_idx+1)*args.chunk_size]
    total_batches = 5 if debug else len(keys)

    if not tokenizer.convert_tokens_to_ids("<|eot_id|>"):
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|eot_id|>"]})
        model.resize_token_embeddings(len(tokenizer))

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    for i in tqdm(range(args.start_idx, total_batches, batch_size), desc="Processing Batches"):
        key = keys[i]
        messages = [
            {
                "role": "user",
                "content": """\\ You are shopping for a given query. Your task to provide the probability of purchase for the entire list, that is, given the list of items in the given order what is the probability to purchase of atleast one item. It should be a single number between 0 and 1. Also select one item that you would like to purchase. \n
        You enter a 'query' into the shopping system and it returns some items mentioned in the 'products'. The items are presented in the given order with 1st item shown in the top of the list and last item shown at the bottom. \n
        Your query_products shopping list: "{query_group}"

        Relevance Score: The relevance score shows how relevant is the item given the query. For every query-item pair it is a numerical value between 0 and 1. Compute the relevance score for each product based on how relevant the product is for the given query. \n
        You should consider other criterial such as:\n
        1. Position bias: where the items appearing near the top are likely to be clicked.
            The position score decrease following the position probabilites: position_scores = {{
                1: 1.0,
                2: 0.6737794280052185,
                3: 0.4144741892814636,
                4: 0.29320451617240906,
                5: 0.20786237716674805,
                6: 0.17144884169101715,
                7: 0.13630738854408264,
                8: 0.11656076461076736,
                9: 0.08377900719642639,
                10: 0.05790691450238228,
                11: 0.05269654467701912,
                12: 0.04374216869473457,
                13: 0.03947276994585991,
                14: 0.028918657451868057,
                15: 0.03581502288579941
            }} \n
            If the relevant item is not near the top it will reduce the probability of purchase irrespective of relevance. \n
        2. Brand bias: If items from similar brand are placed adjacent to each to then then it would discourgae the user from making any purchase. The user will mostly not purchase any item from same brand. High brand bias means items from same brand are adjacent, while low brand bias means items from differnet brands are present. \n
        3. Irrelevance Bias: It is calcuated as the contextual dissimilarity among query-item pairs among the top positions. If mutiple irrelevant items are in the top of list then the liklihood of purhcasing any item deceases, which means irrelevance bias is high. \n
        4. Color Bias: If products with similar colors are placed together, it decreases the liklihood of purchase as there is very less diversity and all products look similar. If many similar colored items are placed together, then the liklihood of purchase of any item decreases, which means color bias is high. \n
        **Note** High brand_bias, high irrelvance_bias, or high color_bias is bad for user, as these will decrease the probability of purchasing any item for the given list. \n
        Task: {{ Estimate the final probabilty of purchase for the entire list given "query_products shopping list" (that is, the probability that you will purchase one of the shown item) and report the result (no need to perform exact final calculations): \n
              Final proability of purchase should be estimated considering relevance scores, position bias, brand bias, irrelevance bias, and color bias. }} \n 
        **Important** Make sure the final line of the ouput should be in following format and initiate the response with "<think>\\n" at the beginning of every output: \n
            Output: P(purchase) = <answer> \n
            Item to be purchased = <answer> \n
 """
            }
        ]

        QG = query_groups[key]
        messages[0]['content'] = messages[0]['content'].format(query_group=QG)

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                eos_token_id=terminators,
                do_sample=False,
                temperature=temperature,
                use_cache=True
            )

        response = outputs[0]
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)

        if debug:
            with open('batch_output.txt', 'a') as file:
                file.write(decoded_response + "\n" + f"{'-'*200}\n" * 2 + "\n")

        probs, items = extract_info_batch(decoded_response)

        if debug:
            print(f"\nQuery : {QG}")
            print(f"Final Probability: {probs}")
            print(f"Item to be Purchased: {items}")

        if probs is not None:
            all_records.append({
                "query_id": QG["query_id"],
                "query": QG["query"],
                "products": json.dumps(QG["products"]),
                "probability": probs,
                "item_selected": items
            })

        if i > 0 and i % args.save_freq == 0:
            output_path = args.data_path.replace(".parquet", f"_qg_predictions_{args.file_idx}_{i}.parquet")
            df_result = pd.DataFrame(all_records)
            df_result.to_parquet(output_path, index=False)

    return pd.DataFrame(all_records)

def run(args):
    """Main execution function that loads model, data, and runs inference.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    start_time = time.time()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()

    model, tokenizer = load_model_and_tokenizer(os.path.join(args.model_dir, args.model_id))
    query_groups = pd.read_parquet(args.data_path)

    print(f"Total query groups: {len(query_groups.keys())} | Batch size: {args.batch_size}")
    df_result = generate_predictions(model, tokenizer, query_groups,
                args.batch_size, args.max_tokens, args.temperature, args.debug, args)

    output_path = args.data_path.replace(".parquet", f"_qg_predictions_{args.file_idx}.parquet")
    df_result.to_parquet(output_path, index=False)

    print(f"Saved query group-level predictions to {output_path}")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

def process_esci(args):
    df_examples = pd.read_parquet(args.esci_examples)
    df_products = pd.read_parquet(args.esci_products)
    df_sources = pd.read_csv(args.sources)

    df_examples_products = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=['product_locale','product_id'],
        right_on=['product_locale', 'product_id']
    )

    df = df_examples_products[df_examples_products["product_locale"] == "us"]

    # df_sampled = (
    #     df.groupby('query_id', group_keys=False)
    #     .apply(lambda x: x.sample(n=min(len(x), args.max_qg_size), random_state=args.random_state))
    # )

    # Load the model
    lang_model = fasttext.load_model(os.path.join(args.data_path,"lid.176.bin"))

    # Fix: Safer version of fasttext language detection
    def fasttext_detect(text):
        try:
            labels, _ = lang_model.predict(text.replace("\n", " ").strip(), k=1)
            return labels[0].replace("__label__", "")
        except Exception:
            return "unknown"

    # Apply detection on unique queries
    unique_queries = df[["query_id", "query"]].drop_duplicates()
    unique_queries["query_language"] = unique_queries["query"].apply(fasttext_detect)

    df = df.merge(unique_queries[["query_id", "query_language"]], on="query_id", how="left")

    # Filter for English queries
    df_en = df[df["query_language"] == "en"]
    df_sampled = get_query_groups(df_en, num_items=args.max_qg_size, random_sampling=not(args.irr_sampling))
    df_sampled.to_parquet(args.data_path+"df_sampled_en.parquet", index=False)

    keys = list(df_sampled.keys())

    with open(args.data_path+"query_group_input.jsonl", "w") as f:
        for qid in keys:
            entry = {"query_group_input": df_sampled[qid]}
            f.write(json.dumps(entry) + "\n")

    #print(f" JSONL saved to: {output_path}")

def main():
    """Parses command-line arguments and initiates the run process."""
    parser = argparse.ArgumentParser(description="Run shopping purchase predictor")
    parser.add_argument("--model_dir", type=str, default="/home/gbhatt/scratch/model_weights/", help="Path to model")
    parser.add_argument("--model_id", type=str, default="DeepSeek-R1-Distill-Llama-8B", help="Path to model")
    parser.add_argument("--root_dir", type=str, default="/ubc/cs/home/g/gbhatt/borg/ranking", help="Path to save processed esci data")
    parser.add_argument("--data_path", type=str, default="/ubc/cs/home/g/gbhatt/borg/ranking/data/esci_data/", help="Path to save processed esci data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--process_esci', action='store_true')
    parser.add_argument('--llm_inf', action='store_true')
    parser.add_argument('--irr_sampling', action='store_true')
    parser.add_argument("--file_idx", type=int, default=0, help="data extraction offset")
    parser.add_argument("--random_state", type=int, default=42, help="data extraction offset")
    parser.add_argument("--max_qg_size", type=int, default=8, help="number of items in QGs")
    parser.add_argument("--start_idx", type=int, default=0, help="manual start of inference")
    parser.add_argument("--chunk_size", type=int, default=2500, help="data extraction size")
    parser.add_argument("--save_freq", type=int, default=200, help="save freq for data extraction")
    parser.add_argument("--esci_examples", type=str, default="'../data/esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet'", help="Path to model")
    parser.add_argument("--esci_products", type=str, default="../data/esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet", help="Path to model")
    parser.add_argument("--esci_sources", type=str, default="../data/esci-data/shopping_queries_dataset/shopping_queries_dataset_sources.csv", help="Path to model")
    args = parser.parse_args()
    
    if args.process_esci:
        process_esci(args)
    
    if args.llm_inf:
        run(args)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")