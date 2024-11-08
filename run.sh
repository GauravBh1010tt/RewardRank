model=rr_ips_hard_pre
ultr_mod='ips'

model_reward=ultr_ips_fix_new
model_ranker=ranker_ips_fix_new

debug=1
train=1
eval=0

epochs=26
lr_drop=18
lr=2e-5
n_viz=1
batch_size=512
n_gpus=2
soft_base=0.8
soft_gain=0.05
output_path=/home/ec2-user/workspace/outputs/
data_path=/home/ec2-user/workspace/data/custom_click
load_path="${output_path}${model}/checkpoints/checkpoint25.pth"
load_path_reward="${output_path}${model_reward}/checkpoints/checkpoint25.pth"
load_path_ranker="${output_path}${model_ranker}/checkpoints/checkpoint25.pth"

if [[ $debug -gt 0 ]]
then
echo "Debugging... "${model}

train=0
eval=0
n_gpus=1

CUDA_VISIBLE_DEVICES=1 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --use_doc_feat --delta_retain=0.5 --soft_base=$soft_base --soft_gain=$soft_gain \
    --n_viz=0 --ultr_mod=$ultr_mod --eval --train_ranker --eval_ultr \
    --load_path=$load_path
    #--train_ranker --eval eval_rels --max_positions_PE=150 --use_dcg --debug --eval_ultr

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
#     --save_epochs=5 --epochs=$epochs --lr_drop=$lr_drop --ultr_models=$ultr_mod \
#     --debug --use_doc_feat --delta_retain=0.5 --soft_base=$soft_base --soft_gain=$soft_gain \
#     --n_viz=5 --eval --force_tnse --perturbation_sampling 
fi

if [[ $train -gt 0 ]]
then
echo "Training... "${model}

CUDA_VISIBLE_DEVICES=4,5 python main.py --n_gpus=$n_gpus \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=4 \
    --n_viz=0  --use_doc_feat --ultr_models=$ultr_mod --train_ranker --load_path_reward=$load_path_reward \
    --pretrain_ranker --load_path_ranker=$load_path_ranker --cls_reg --ste
    # --train_ranker_lambda --load_path_reward=$load_path_reward --ste

# CUDA_VISIBLE_DEVICES=5 python main.py --n_gpus=1 --load_path_reward=$load_path_reward --load_path=$load_path \
#     --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
#     --num_workers=4  --n_viz=0 --train_ranker --train_ranker_lambda \
#     --eval --eval_rels --max_positions_PE=150 --use_dcg --concat_feats

fi

if [[ $eval -gt 0 ]]
then
model=ultr_ips
n_gpus=1
batch_size=1024
n_viz=10000
echo "Evaluating... "${model}
load_path="${output_path}${model}/checkpoints/checkpoint25.pth"

CUDA_VISIBLE_DEVICES=0 python main.py --n_gpus=$n_gpus \
    --output_path=$output_path --output_folder=$model --load_path=$load_path \
    --batch_size=$batch_size --eval --use_doc_feat --save_cls --n_viz=$n_viz --ultr_models=$ultr_mod

wait

CUDA_VISIBLE_DEVICES=0 python main.py --n_gpus=$n_gpus \
    --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --force_tnse \
    --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --delta_retain=0.8 --save_fname=per_0.2 --ultr_models=$ultr_mod &

CUDA_VISIBLE_DEVICES=1 python main.py --n_gpus=$n_gpus \
    --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --force_tnse \
    --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_bot --save_fname=swap_bot --ultr_models=$ultr_mod &

CUDA_VISIBLE_DEVICES=2 python main.py --n_gpus=$n_gpus \
    --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --force_tnse \
    --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_top --save_fname=swap_top --ultr_models=$ultr_mod &

CUDA_VISIBLE_DEVICES=3 python main.py --n_gpus=$n_gpus \
    --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --force_tnse \
    --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_rand --save_fname=swap_rand --ultr_models=$ultr_mod &

wait

python src/utils.py --output_path=$output_path --output_folder=$model
fi


# if [[ $eval -gt 0 ]]
# then
# # model=sr_df
# n_viz=100
# echo "Evaluating... "${model}
# load_path="${output_path}${model}/checkpoints/checkpoint35.pth"

# CUDA_VISIBLE_DEVICES=4 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path \
#     --batch_size=$batch_size --eval --use_doc_feat --save_cls --n_viz=$n_viz --ultr_models=$ultr_mod

# CUDA_VISIBLE_DEVICES=4 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling \
#     --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --delta_retain=0.8 --save_fname=per_0.2 --ultr_models=$ultr_mod &

# CUDA_VISIBLE_DEVICES=5 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling \
#     --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_bot --save_fname=swap_bot --ultr_models=$ultr_mod &

# CUDA_VISIBLE_DEVICES=6 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling \
#     --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_top --save_fname=swap_top --ultr_models=$ultr_mod &

# sleep 120

# CUDA_VISIBLE_DEVICES=7 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --merge_imgs \
#     --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_rand --save_fname=swap_rand --ultr_models=$ultr_mod &
# fi