
model=dfeat_un
output_path=/home/ggbhatt/workspace/cf_ranking/outputs/
load_path="${output_path}${model}/checkpoint04.pth"
problem_type="classification"
debug=1
train=1
eval=1
lr=2e-5
wt_decay=1e-2
epochs=40
lr_drop=20

if [[ $debug -gt 0 ]]
then
echo "Debugging..."

train=0
eval=0

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --batch_size=256 --output_path=$output_path --output_folder=$model --problem_type=${problem_type} \
    --save_epochs=5 --lr=$lr --weight_decay=$wt_decay --epochs=$epochs --lr_drop=$lr_drop \
    --debug --use_doc_feat
fi

if [[ $train -gt 0 ]]
then
echo "Training..."

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --batch_size=256 --output_path=$output_path --output_folder=$model --problem_type=${problem_type} \
    --save_epochs=5 --lr=$lr --weight_decay=$wt_decay --epochs=$epochs --lr_drop=$lr_drop \
    --use_doc_feat
fi

if [[ $eval -gt 0 ]]
then
echo "Evaluating..."
load_path="${output_path}${model}/checkpoint35.pth"

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --output_path=$output_path --output_folder=$model --load_path=$load_path --eval
fi