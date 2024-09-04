
model=lr4_d2

load_path="/home/ggbhatt/workspace/cf_ranking/outputs/${model}/checkpoint04.pth"

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --batch_size=256 --output_folder=$model --problem_type="classification" \
    --save_epochs=5 --use_wandb --lr=2e-4 --weight_decay=1e-2 --epochs=10 \
    --load_path=$load_path --eval