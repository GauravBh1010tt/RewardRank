
load_path="/home/ggbhatt/workspace/cf_ranking/outputs/CE_b256/checkpoint01.pth"
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --batch_size=256 --output_folder=CE_b256 --problem_type="classification" \
    --save_epochs=5 --use_wandb --lr=2e-4 --weight_decay=1e-2 --epochs=10 \
    --load_path=$load_path --eval

rm -rf wandb