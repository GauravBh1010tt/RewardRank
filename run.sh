
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --batch_size=256 --output_folder=reg_b256 --problem_type="regression" \
    --use_wandb