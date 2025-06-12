ultr_mod='ips'
model_reward=reward_ips
model=rr_ips_hard_corr_0.0
model_ranker=ranker_ips

ckpt=20

debug=0
ips_train=0
llm_train=0
org_train=0
ips_eval=0
llm_eval=1

epochs=21
lr_drop=12
lr=2e-5
n_viz=0
batch_size=256
n_gpus=3
soft_base=0.8
soft_gain=0.05
output_path=/ubc/cs/home/g/gbhatt/borg/ranking/outputs/
data_path=/ubc/cs/home/g/gbhatt/borg/ranking/data/custom_click
#data_path=/ubc/cs/home/g/gbhatt/borg/ranking/data/llm_data/processed
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

CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --delta_retain=0.5 --soft_base=$soft_base --soft_gain=$soft_gain --n_viz=0 \
    --train_ranker --load_path_reward=$load_path_reward --ultr_models=$ultr_mod --use_doc_feat \
    --debug
fi 

if [[ $ips_train -gt 0 ]]
then
echo "IPS Training... "${model}
data_path=/ubc/cs/home/g/gbhatt/borg/ranking/data/custom_click

CUDA_VISIBLE_DEVICES=2,3,4 python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=4 \
    --n_viz=0 --train_ranker --load_path_reward=$load_path_reward --ultr_models=$ultr_mod --use_doc_feat \
    --residual_coef=0.5 --reward_correction --ste
fi

if [[ $llm_train -gt 0 ]]
then
echo "LLM Training... "${model}
data_path=/ubc/cs/home/g/gbhatt/borg/ranking/data/llm_data/processed

CUDA_VISIBLE_DEVICES=7 python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=4 \
    --n_viz=0 --train_ranker --load_path_reward=$load_path_reward  \
    --residual_coef=1.5 --reward_correction --llm_exp

fi

if [[ $ips_eval -gt 0 ]]
then
echo "IPS Eval... "${model}
data_path=/ubc/cs/home/g/gbhatt/borg/ranking/data/custom_click

batch_size=256
n_gpus=6

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --delta_retain=0.5 --soft_base=$soft_base --soft_gain=$soft_gain --n_viz=0 \
    --load_path_reward=$load_path_reward --gain_fn=exp --train_ranker --use_doc_feat \
    --load_path=$load_path --eval --eval_ultr 
fi


if [[ $llm_eval -gt 0 ]]
then
model=ranker_llm
echo "LLM Eval... "${model}
data_path=/ubc/cs/home/g/gbhatt/borg/ranking/data/llm_data/processed
load_path="${output_path}${model}/checkpoints/checkpoint${ckpt}.pth"
n_gpus=1

CUDA_VISIBLE_DEVICES=0 python eval_llm.py \
    --batch_size=$batch_size --output_path=$output_path --data_path=$data_path \
    --n_gpus=$n_gpus --load_path=$load_path --eval_llm --llm_exp \
    --output_folder=$model --eval_online --train_ranker

fi

if [[ $org_train -gt 0 ]]
then

batch_size=256
n_gpus=1
ckpt=20

model=ranker_org

echo "Org Training... "${model}
data_path=/ubc/cs/home/g/gbhatt/borg/ranking/data/llm_data/processed
model_reward=reward_org

load_path_reward="${output_path}${model_reward}/checkpoints/checkpoint25.pth"
load_path="${output_path}${model}/checkpoints/checkpoint${ckpt}.pth"

CUDA_VISIBLE_DEVICES=0 python main.py --n_gpus=$n_gpus \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=10 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=4 \
    --n_viz=0 --train_ranker --load_path_reward=$load_path_reward  \
    --residual_coef=1.0 --reward_correction --use_org_feats --use_doc_feat --load_path=$load_path --eval
fi