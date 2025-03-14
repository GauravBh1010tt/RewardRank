model=choice1_pgrank_mc1_t0
ultr_mod='ips'

model_reward=reward_ips_choice1
model_ranker=ranker_ips

debug=0
train=0
eval=1

epochs=21
lr_drop=12
lr=2e-5
n_viz=0
batch_size=341
n_gpus=3
soft_base=0.8
soft_gain=0.05
output_path=/ubc/cs/home/g/gbhatt/borg/ranking/outputs/
data_path=/ubc/cs/home/g/gbhatt/borg/ranking/data/custom_click
load_path="${output_path}${model}/checkpoints/checkpoint20.pth"
load_path_reward="${output_path}${model_reward}/checkpoints/checkpoint20.pth"
load_path_ranker="${output_path}${model_ranker}/checkpoints/checkpoint20.pth"

if [[ $debug -gt 0 ]]
then
echo "Debugging... "${model}

train=0
eval=0
n_gpus=1
batch_size=8

CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --use_doc_feat --delta_retain=0.5 --soft_base=$soft_base --soft_gain=$soft_gain --n_viz=0 \
    --eval --eval_ultr --debug --train_ranker --load_path=$load_path
    # --load_path_reward=$load_path_reward --ultr_models=$ultr_mod --gain_fn=exp --train_ranker \
    # --load_path=$load_path --eval --eval_ultr --n_viz=0 #--pgrank_loss --pgrank_disc
    # --pgrank_loss --pgrank_disc --debug
    # --pgrank_loss --pgrank_disc --debug --MC_samples=2
    #--load_path=$load_path --ultr_models=$ultr_mod --eval --eval_ultr --gain_fn=exp --train_ranker --n_viz=0 --MC_samples=2
    #--load_path=$load_path --eval --eval_ultr --gain_fn=exp --debug
fi

if [[ $train -gt 0 ]]
then
echo "Training... "${model}

CUDA_VISIBLE_DEVICES=5,6,7 python main.py --n_gpus=$n_gpus --use_wandb \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --epochs=$epochs --lr=$lr --lr_drop=$lr_drop --num_workers=2 --eval_epochs=4 \
    --n_viz=0  --use_doc_feat --ultr_models=$ultr_mod --train_ranker --load_path_reward=$load_path_reward \
    --pgrank_loss --pgrank_disc --MC_samples=10 --gain_fn=exp
    # --pgrank_loss --pgrank_disc --MC_samples=10 --gain_fn=exp
    # --use_soft_perm_loss --soft_perm_loss_reg=0.1 
    # --pretrain_ranker --load_path_ranker=$load_path_ranker --urcc_loss --per_item_feats
    # --cls_reg --ste
    #--pretrain_ranker --load_path_ranker=$load_path_ranker --cls_reg --ste
    # --train_ranker_lambda --load_path_reward=$load_path_reward --ste --ultr_models=$ultr_mod

# CUDA_VISIBLE_DEVICES=5 python main.py --n_gpus=1 --load_path_reward=$load_path_reward --load_path=$load_path \
#     --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
#     --num_workers=4  --n_viz=0 --train_ranker --train_ranker_lambda \
#     --eval --eval_rels --max_positions_PE=150 --use_dcg --concat_feats

fi

if [[ $eval -gt 0 ]]
then
echo "Eval... "${model}

CUDA_VISIBLE_DEVICES=5,6,7 python main.py \
    --batch_size=$batch_size --output_path=$output_path --output_folder=$model --data_path=$data_path \
    --save_epochs=5 --eval_epochs=1 --epochs=$epochs --lr_drop=$lr_drop --n_gpus=$n_gpus \
    --use_doc_feat --delta_retain=0.5 --soft_base=$soft_base --soft_gain=$soft_gain --n_viz=0 \
    --load_path_reward=$load_path_reward --ultr_models=$ultr_mod --gain_fn=exp --train_ranker \
    --load_path=$load_path --eval --eval_ultr
    # --pgrank_loss --pgrank_disc --debug
    # --pgrank_loss --pgrank_disc --debug --MC_samples=2
    #--load_path=$load_path --ultr_models=$ultr_mod --eval --eval_ultr --gain_fn=exp --train_ranker --n_viz=0 --MC_samples=2
    #--load_path=$load_path --eval --eval_ultr --gain_fn=exp --debug

# CUDA_VISIBLE_DEVICES=0 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path \
#     --batch_size=$batch_size --eval --use_doc_feat --save_cls --n_viz=$n_viz --ultr_models=$ultr_mod

# wait

# CUDA_VISIBLE_DEVICES=0 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --force_tnse \
#     --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --delta_retain=0.8 --save_fname=per_0.2 --ultr_models=$ultr_mod &

# CUDA_VISIBLE_DEVICES=1 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --force_tnse \
#     --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_bot --save_fname=swap_bot --ultr_models=$ultr_mod &

# CUDA_VISIBLE_DEVICES=2 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --force_tnse \
#     --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_top --save_fname=swap_top --ultr_models=$ultr_mod &

# CUDA_VISIBLE_DEVICES=3 python main.py --n_gpus=$n_gpus \
#     --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --force_tnse \
#     --batch_size=$batch_size --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_rand --save_fname=swap_rand --ultr_models=$ultr_mod &

# wait

# python src/utils.py --output_path=$output_path --output_folder=$model
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