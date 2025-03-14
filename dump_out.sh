split=test
out_dir=/ubc/cs/home/g/gbhatt/borg/ranking/data/custom_click
data_root=/ubc/cs/home/g/gbhatt/borg/ranking/data/philipphager___baidu-ultr_uva-mlm-ctr/clicks/0.1.0/60cc071890b9bcc27adbfc78a642f1fa5d1668d90fadbe5b9fedcf3cd37bc89f/
org=0

# CUDA_VISIBLE_DEVICES=6 python create_out.py --split=$split --part=0 --out_dir=$out_dir --data_root=$data_root

if [[ $org -gt 0 ]]
then
echo "split:: "${split}
CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=0 --out_dir=$out_dir --data_root=$data_root &
CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=7 --out_dir=$out_dir --data_root=$data_root &
CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=14 --out_dir=$out_dir --data_root=$data_root &
CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=21 --out_dir=$out_dir --data_root=$data_root &
CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=28 --out_dir=$out_dir --data_root=$data_root &
CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=35 --out_dir=$out_dir --data_root=$data_root &
CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=42 --out_dir=$out_dir --data_root=$data_root &
CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=49 --out_dir=$out_dir --data_root=$data_root &
else
echo "split:: "${split}
# CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=0 --out_dir=$out_dir --data_root=$data_root &
# CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=3 --out_dir=$out_dir --data_root=$data_root &
# CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=6 --out_dir=$out_dir --data_root=$data_root &
# CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=9 --out_dir=$out_dir --data_root=$data_root &
CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=12 --out_dir=$out_dir --data_root=$data_root &
CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=15 --out_dir=$out_dir --data_root=$data_root &



# CUDA_VISIBLE_DEVICES=6 python create_out_solo.py --split=$split --part=18 --st_part=$missing_part --out_dir=$out_dir &
# CUDA_VISIBLE_DEVICES=7 python create_out_solo.py --split=$split --part=21 --st_part=$missing_part --out_dir=$out_dir &

#wait
#echo "concatenating temp files"
#CUDA_VISIBLE_DEVICES=7 python create_out_solo.py --split=$split --st_part=$missing_part --concat --out_dir=$out_dir
fi