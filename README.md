# RewardRank

## Outline

- [RewardRank](#rewardrank)
  - [Outline](#outline)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
    - [PO-Eval](#kd-eval)
    - [LAU-Eval](#lau-eval)
  - [Training reward model](#training-reward-model)
  - [Training rankers](#training-rankers)
  - [Evaluation](#evaluation)
  - [Citation](#citation)


## Installation

To create the Conda environment from the `env.yaml` file:

1. Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.
2. Run the following command in your terminal:

   ```bash
   # Replace 'my_env_name' with your desired environment name
   # Replace '/your/custom/path' with the path where you want to create the environment

   $conda env create --file env.yaml --name my_env_name --prefix /your/custom/path
    ```
## Data Preparation
### PO-Eval
We use the dataset provided by [ULTR-reranking (arXiv:2404.02543)](https://arxiv.org/pdf/2404.02543), which can be downloaded from [HuggingFace](https://huggingface.co/datasets/philipphager/baidu-ultr_uva-mlm-ctr).

By default, the dataset is cached in the `.cache` directory. We use this dataset and distill soft labels using the IPS model from the [baidu-bert-model repository](https://github.com/philipphager/baidu-bert-model).

To set up the IPS model:

```bash
$ cd RewardRank
$ git clone git@github.com:philipphager/baidu-bert-model.git
$ mv baidu-bert-model bbm
```

We will use `src/create_kdeval_data.py` to generete the pseduo labels. Pass the path of downloaded data (by default in the `~/.cache/huggingface/philipphager___baidu-ultr_uva-mlm-ctr/clicks/0.1.0/...`). Also provide the path of the output folder.

```bash
$ python src/create_kdeval_data.py --data_root='' --out_dir=''
```
---
### LAU-Eval
Download the [KDD Cup dataset](https://arxiv.org/pdf/2206.06588) from the [official ESCI GitHub repository](https://github.com/amazon-science/esci-data). The dataset consists of the following three files:

- `shopping_queries_dataset_examples.parquet`
- `shopping_queries_dataset_products.parquet`
- `shopping_queries_dataset_sources.csv`

We preprocess the dataset by filtering out non-English query groups (QGs) and sampling 8 items per QG.

To run the preprocessing script, provide the paths to the `.parquet` and `.csv` files and use the `--process_esci` flag:

```bash
$ python src/create_llm_data.py --process_esci \
    --esci_examples='path/to/shopping_queries_dataset_examples.parquet' \
    --esci_products='path/to/shopping_queries_dataset_products.parquet' \
    --esci_sources='path/to/shopping_queries_dataset_sources.csv'
```

To estimate `purchase_probability` using an LLM, we assume the model from Hugging Face has been downloaded locally at the path `model_dir/model_id`. You can configure inference parameters such as maximum context length (`--max_tokens`) and temperature as needed.

Run the inference script with the following command:

```bash
$ python src/create_llm_data.py --llm_inf \
    --model_dir='path/to/model_dir' \
    --model_id='model_id' \
    --max_tokens=4096
```

For Claude 3.5 Sonnet v2, we perform batch inference using [Amazon Bedrock](https://aws.amazon.com/bedrock/anthropic/?ams%23interactive-card-vertical%23pattern-data.filter=%257B%2522filters%2522%253A%255B%255D%257D). Similar batch inference capabilities are also available for GPT-4o and other commercial LLMs through their respective APIs.

## Training reward model
To train the reward model, first configure the appropriate flags in the `train_reward.sh` script based on your evaluation strategy. Specify the correct path to the preprocessed dataset.

### For PO-Eval  
Set the following flag:
```bash
ips_train=1
```
---
### For LAU-Eval  
Set the following flags:
```bash
llm_train=1
ips_train=0
org_train=0
```

You can adjust other training parameters (e.g., learning rate, batch size, n_gpus, etc) within the `train_reward.sh` script as needed. Once configured, run the script to start training:
```bash
$ bash train_reward.sh
```

## Training rankers
To train the rankers, first configure the appropriate flags in the `train_ranker.sh` script based on your evaluation strategy (by setting `ips_train` and `llm_train` flag). Specify the correct path to the preprocessed dataset.
```bash
$ bash train_ranker.sh
```
### RewardRank Training
Use the following flags (to be run using `train_ranker.sh`) to train `RewardRank`:
```bash
python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path \
    --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop \
    --num_workers=2 --eval_epochs=4 --train_ranker \
    --load_path_reward=$load_path_reward --ultr_models=$ultr_mod \
    --use_doc_feat --residual_coef=0.5 --reward_correction
```

---

### PG-RANK* Training
Use the following flags (to be run using `train_ranker.sh`) to train `PG-RANK*`:
```bash
python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path \
    --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop \
    --num_workers=2 --eval_epochs=4 --train_ranker \
    --load_path_reward=$load_path_reward --ultr_models=$ultr_mod \
    --use_doc_feat --pgrank_loss --pgrank_disc -mc_sample=5
```

---

### URCC Training
Use the following flags (to be run using `train_ranker.sh`) to train `URCC*`:
```bash
python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path \
    --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop \
    --num_workers=2 --eval_epochs=4 --train_ranker \
    --load_path_reward=$load_path_reward --ultr_models=$ultr_mod \
    --use_doc_feat --urcc_loss
```

Be sure to set the appropriate flags for `LLM_exp` in the `train_ranker.sh` script.

## Evaluation
To evaluate the rankers, first configure the appropriate flags in the `eval_ranker.sh` script based on your evaluation strategy (by setting `ips_eval` and `llm_eval` flag). Specify the correct path to the preprocessed dataset.
```bash
$ bash eval_ranker.sh
```
### PO_Eval
Evaluate the trained model using `PO_Eval` by using the following flags (to be run using `eval_ranker.sh`):
```bash
python main.py --batch_size=$batch_size --output_path=$output_path \
    --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs \
    --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --load_path_reward=$load_path_reward --gain_fn=exp \
    --train_ranker --use_doc_feat --load_path=$load_path \
    --eval --eval_ultr
```
---

### LAU_Eval (Online)
Generate inference-based ranking data for online LLM evaluation by using the following flags (to be run using `eval_ranker.sh`):
```bash
python eval_llm.py --batch_size=$batch_size --output_path=$output_path \
    --data_path=$data_path --n_gpus=$n_gpus --load_path=$load_path \
    --eval_llm --llm_exp --output_folder=$model \
    --eval_online --train_ranker
```

---

### LAU_Eval (Offline)
Evaluate LLM responses for a given ranking strategy by using the following flags (to be run using `eval_ranker.sh`):
```bash
python eval_llm.py --batch_size=$batch_size --output_path=$output_path \
    --data_path=$data_path --n_gpus=$n_gpus --load_path=$load_path \
    --eval_llm --llm_exp --output_folder=$model \
    --eval_offline --train_ranker
```
Don’t forget to set the required flags and paths in the `eval_ranker.sh` script before running evaluations.

<!--
## Citation
If you find this repo useful, please cite:
```
@misc{bhatt2025rewardrank,
      title={RewardRank: Optimizing True Learning-to-Rank Utility}, 
      author={Gaurav Bhatt and Kiran Koshy Thekumparampil and Tanmay Gangwani and Tesi Xiao and Leonid Sigal},
      year={2025},
      eprint={2508.14180},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2508.14180}, 
}
```
-->
