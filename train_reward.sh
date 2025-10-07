ultr_mod='ips'

model=reward_cls_item
# model_reward=reward_ips_org_pos
model_reward=$model
exp_type=p2_llm_t0.5
ckpt=20
base_dir=/ubc/cs/home/g/gbhatt/borg/ranking

debug=0
ips_train=0
llm_train=1
org_train=0
ips_eval=0
llm_eval=0

epochs=21
lr_drop=12
lr=2e-5
n_viz=0
batch_size=256
n_gpus=3
# soft_base=0.8
# soft_gain=0.05
output_path=${base_dir}/outputs/${exp_type}/
data_path=${base_dir}/data/custom_click
#data_path=/ubc/cs/home/g/gbhatt/borg/ranking/data/llm_data/processed
load_path="${output_path}${model}/checkpoints/checkpoint${ckpt}.pth"
load_path_reward="${output_path}${model_reward}/checkpoints/checkpoint${ckpt}.pth"
load_path_ranker="${output_path}${model_ranker}/checkpoints/checkpoint${ckpt}.pth"

if [[ $debug -gt 0 ]]
then
echo "Debugging... "${model}

n_gpus=1
batch_size=10
data_path=${base_dir}/data/custom_click

CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=4 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --n_viz=0 --debug --use_doc_feat --per_item_feats --reward_loss_cls --use_org_feats --use_dcg
    # --ultr_models=$ultr_mod
fi 

if [[ $ips_train -gt 0 ]]
then
echo "IPS Reward Training...   exp: "${exp_type}"    model: "${model}
data_path=${base_dir}/data/custom_click

CUDA_VISIBLE_DEVICES=2,3,4 python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=4 \
    --use_doc_feat --reward_loss_cls --per_item_feats --ips_sampling --ultr_models=$ultr_mod
fi

if [[ $llm_train -gt 0 ]]
then

batch_size=256
n_gpus=2
echo "LLM Reward Training...   exp: "${exp_type}"    model: "${model}
data_path=${base_dir}/data/${exp_type}/processed

CUDA_VISIBLE_DEVICES=4,5 python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=4 \
    --n_viz=0 --llm_exp --reward_loss_cls --per_item_feats
fi

if [[ $ips_eval -gt 0 ]]
then
echo "IPS Eval... "${model}
data_path=${base_dir}/data/custom_click

batch_size=256
n_gpus=6

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --n_viz=0 --load_path_reward=$load_path_reward --gain_fn=exp --use_doc_feat --ultr_models=$ultr_mod \
    --eval --reward_sanity --save_soft_labels --reward_correction --residual_coef=1.0
fi

if [[ $llm_eval -gt 0 ]]
then
#model=rr_llm_pgrank_mc5
echo "LLM Eval... "${model}
data_path=${base_dir}/data/${exp_type}/processed
n_gpus=1

CUDA_VISIBLE_DEVICES=0 python main.py --n_gpus=$n_gpus \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=4 \
    --n_viz=0 --llm_exp --load_path_reward=$load_path_reward --eval --reward_sanity --eval_llm \
    --save_soft_labels --reward_correction --residual_coef=1.0
fi

if [[ $org_train -gt 0 ]]
then

echo "Org Reward Training...   exp: "${exp_type}"    model: "${model}

CUDA_VISIBLE_DEVICES=2,3,4 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=4 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus --num_workers=2 \
    --n_viz=0 --use_doc_feat --per_item_feats --reward_loss_cls --use_org_feats --use_dcg
fi