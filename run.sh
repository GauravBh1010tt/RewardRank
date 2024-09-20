model=soft_0.8_0.05
output_path=/home/ggbhatt/workspace/cf_ranking/outputs/
load_path="${output_path}${model}/checkpoint35.pth"
problem_type="classification"
debug=0
train=1
eval=0
lr=2e-5
wt_decay=1e-2
epochs=40
lr_drop=20

if [[ $debug -gt 0 ]]
then
echo "Debugging... "${model}

train=0
eval=0

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --batch_size=256 --output_path=$output_path --output_folder=$model --problem_type=${problem_type} \
    --save_epochs=5 --lr=$lr --weight_decay=$wt_decay --epochs=$epochs --lr_drop=$lr_drop \
    --debug --use_doc_feat --save_cls --delta_retain=1.0 --soft_labels --soft_base=0.9 --soft_gain=0.02
fi

if [[ $train -gt 0 ]]
then
echo "Training... "${model}

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --batch_size=256 --output_path=$output_path --output_folder=$model --problem_type=${problem_type} \
    --save_epochs=5 --lr=$lr --weight_decay=$wt_decay --epochs=$epochs --lr_drop=$lr_drop \
    --use_doc_feat --save_cls --soft_labels --soft_base=0.8 --soft_gain=0.05
fi

if [[ $eval -gt 0 ]]
then
echo "Evaluating... "${model}
load_path="${output_path}${model}/checkpoint35.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --output_path=$output_path --output_folder=$model --load_path=$load_path \
    --batch_size=1024 --eval --use_doc_feat --save_cls --n_viz=5 --delta_retain=0.1 --save_fname=per_0.9
fi