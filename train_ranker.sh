ultr_mod='ips'

model_reward=reward_cls_item
model=pgrank_mc10
model_ranker=ranker_ips
base_dir=/ubc/cs/home/g/gbhatt/borg/ranking

exp_type=ips_bin
ckpt=20

debug=0
ips_train=1
llm_train=0
org_train=0

epochs=12
lr_drop=12
lr=2e-5

batch_size=256
n_gpus=3

output_path=${base_dir}/outputs/${exp_type}/
data_path=${base_dir}/data/custom_click
load_path="${output_path}${model}/checkpoints/checkpoint${ckpt}.pth"
load_path_reward="${output_path}${model_reward}/checkpoints/checkpoint${ckpt}.pth"
load_path_ranker="${output_path}${model_ranker}/checkpoints/checkpoint${ckpt}.pth"

if [[ $debug -gt 0 ]]
then
echo "Debugging... "${model}

train=0
eval=0
n_gpus=1
batch_size=10

data_path=${base_dir}/data/custom_click

CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --delta_retain=0.5 --n_viz=0 \
    --train_ranker --load_path_reward=$load_path_reward --debug --use_doc_feat --po_eval \
    --soft_sort_temp=1.0 --residual_coef=0.0 --reward_correction --grpo_loss
fi 

if [[ $ips_train -gt 0 ]]
then
echo "IPS Training...   exp: "${exp_type}"    model: "${model}
data_path=${base_dir}/data/custom_click

CUDA_VISIBLE_DEVICES=2,3,4 python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=4 \
    --n_viz=0 --train_ranker --load_path_reward=$load_path_reward --po_eval --use_doc_feat \
    --grpo_loss --grpo_rollouts=8
fi

if [[ $llm_train -gt 0 ]]
then
batch_size=512
n_gpus=1
echo "LLM Ranker Training...   exp: "${exp_type}"    model: "${model}
data_path=${base_dir}/data/${exp_type}/processed

CUDA_VISIBLE_DEVICES=6 python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=4 \
    --n_viz=0 --train_ranker --load_path_reward=$load_path_reward  --lau_eval \
    --soft_sort_temp=0.3 --residual_coef=0.6 --reward_correction

CUDA_VISIBLE_DEVICES=6 python eval_llm.py \
    --batch_size=$batch_size --output_path=$output_path --data_path=$data_path \
    --n_gpus=1 --load_path=$load_path --eval_llm --lau_eval --output_folder=$model \
    --eval_online --train_ranker
fi

if [[ $org_train -gt 0 ]]
then

echo "Org Training...   exp: "${exp_type}"    model: "${model}
data_path=${base_dir}/data/custom_click

CUDA_VISIBLE_DEVICES=2,3,4 python main.py --n_gpus=$n_gpus \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=1 \
    --n_viz=0 --train_ranker --use_doc_feat  --use_org_feats --use_dcg --load_path_reward=$load_path_reward \
    --train_ranker_naive --ips_sampling --rank_loss=lambdarank
fi
