
model=pgrank_mc10

ultr_mod='ips'
model_ranker=ranker_ips
model_reward=reward_cls_item
exp_type=ips_bin
base_dir=/ubc/cs/home/g/gbhatt/borg/ranking

ckpt=10

debug=0
ips_eval=1
llm_eval=0
org_eval=0

epochs=10
lr_drop=12
lr=2e-5
n_viz=0
batch_size=256
n_gpus=3
soft_base=0.8
soft_gain=0.05
output_path=${base_dir}/outputs/${exp_type}/
data_path=${base_dir}/data/custom_click
load_path="${output_path}${model}/checkpoints/checkpoint${ckpt}.pth"
load_path_reward="${output_path}${model_reward}/checkpoints/checkpoint20.pth"
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
    --train_ranker --load_path=$load_path --use_doc_feat \
    --debug --eval
fi 

if [[ $ips_eval -gt 0 ]]
then
echo "IPS Eval... "${model}"  checkpoint:"${ckpt}
data_path=${base_dir}/data/custom_click

batch_size=256
n_gpus=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --delta_retain=0.5 --soft_base=$soft_base --soft_gain=$soft_gain --n_viz=0 \
    --gain_fn=exp --train_ranker --use_doc_feat --load_path_reward=$load_path_reward \
    --po_eval --load_path=$load_path --eval --reward_sanity --reward_correction --residual_coef=0.0
fi


if [[ $llm_eval -gt 0 ]]
then
mod=pgrank-mc10
model=${mod}
exp_type=p2_llm_t0.5
echo "LLM Ranker Evaluation...   exp: "${exp_type}"    model: "${model}
data_path=${base_dir}/data/${exp_type}/processed

output_path=${base_dir}/outputs/${exp_type}/
load_path="${output_path}${model}/checkpoints/checkpoint${ckpt}.pth"
n_gpus=1
out_file=${output_path}inf_p2_per/batch_inference_wout-p2-${mod}_0.jsonl.out

CUDA_VISIBLE_DEVICES=2 python eval_llm.py \
    --batch_size=$batch_size --output_path=$output_path --data_path=$data_path \
    --n_gpus=$n_gpus --load_path=$load_path --eval_llm --lau_eval \
    --output_folder=$model --train_ranker --out_file=$out_file #--eval_online 
    #--purchase_prob --eval_online 

fi


if [[ $org_eval -gt 0 ]]
then
echo "Org Eval... "${model}"  checkpoint:"${ckpt}
data_path=${base_dir}/data/custom_click

batch_size=256
n_gpus=4

CUDA_VISIBLE_DEVICES=2,3,4,5 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --delta_retain=0.5 --soft_base=$soft_base --soft_gain=$soft_gain --n_viz=0 \
    --gain_fn=exp --train_ranker --use_doc_feat --use_org_feats --use_dcg \
    --load_path=$load_path --eval --eval_ultr #--reward_sanity \
    #--reward_correction --residual_coef=0.5
fi

