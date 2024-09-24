model=soft_0.8_0.05
output_path=/home/ggbhatt/workspace/cf_ranking/outputs/
load_path="${output_path}${model}/checkpoint35.pth"
problem_type="classification"
debug=0
train=0
eval=1
lr=2e-5
wt_decay=1e-2
epochs=40
lr_drop=20
n_viz=1
soft_base=0.8
soft_gain=0.05

if [[ $debug -gt 0 ]]
then
echo "Debugging... "${model}

train=0
eval=0

CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size=256 --output_path=$output_path --output_folder=$model --problem_type=${problem_type} \
    --save_epochs=5 --lr=$lr --weight_decay=$wt_decay --epochs=$epochs --lr_drop=$lr_drop \
    --debug --use_doc_feat --save_cls --delta_retain=1.0 --soft_base=$soft_base --soft_gain=$soft_gain \
    --eval --n_viz=5 --perturbation_sampling --sampling_type=swap_first_click_bot
fi

if [[ $train -gt 0 ]]
then
echo "Training... "${model}

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --batch_size=256 --output_path=$output_path --output_folder=$model --problem_type=${problem_type} \
    --save_epochs=5 --lr=$lr --weight_decay=$wt_decay --epochs=$epochs --lr_drop=$lr_drop \
    --use_doc_feat --save_cls --soft_labels --soft_base=0.8 --soft_gain=0.05 --perturbation_sampling \
    --sampling_type=swap_first_click_bot
fi

if [[ $eval -gt 0 ]]
then
# model=sr_df
n_viz=5
echo "Evaluating... "${model}
load_path="${output_path}${model}/checkpoint35.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --output_path=$output_path --output_folder=$model --load_path=$load_path \
    --batch_size=1024 --eval --use_doc_feat --save_cls --n_viz=$n_viz --soft_labels --soft_base=$soft_base --soft_gain=$soft_gain

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling \
    --batch_size=1024 --eval --use_doc_feat --n_viz=$n_viz --delta_retain=0.8 --save_fname=per_0.2 --soft_labels --soft_base=$soft_base --soft_gain=$soft_gain

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling \
    --batch_size=1024 --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_bot --save_fname=swap_bot --soft_labels --soft_base=$soft_base --soft_gain=$soft_gain

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling \
    --batch_size=1024 --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_top --save_fname=swap_top --soft_labels --soft_base=$soft_base --soft_gain=$soft_gain

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --output_path=$output_path --output_folder=$model --load_path=$load_path --perturbation_sampling --merge_imgs \
    --batch_size=1024 --eval --use_doc_feat --n_viz=$n_viz --sampling_type=swap_first_click_rand --save_fname=swap_rand --soft_labels --soft_base=$soft_base --soft_gain=$soft_gain
fi