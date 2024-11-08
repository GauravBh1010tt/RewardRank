split=test
out_dir=/home/ec2-user/workspace/data/custom_click_new
org=0

if [[ $org -gt 0 ]]
then
echo "train "${split}
CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=0 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=7 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=2 python create_out.py --split=$split --part=14 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=3 python create_out.py --split=$split --part=21 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=4 python create_out.py --split=$split --part=28 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=5 python create_out.py --split=$split --part=35 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=6 python create_out.py --split=$split --part=42 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=7 python create_out.py --split=$split --part=49 --out_dir=$out_dir &
else
echo "test "${split}
CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=0 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=3 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=2 python create_out.py --split=$split --part=6 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=3 python create_out.py --split=$split --part=9 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=4 python create_out.py --split=$split --part=12 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=5 python create_out.py --split=$split --part=15 --out_dir=$out_dir &
# CUDA_VISIBLE_DEVICES=6 python create_out_solo.py --split=$split --part=18 --st_part=$missing_part --out_dir=$out_dir &
# CUDA_VISIBLE_DEVICES=7 python create_out_solo.py --split=$split --part=21 --st_part=$missing_part --out_dir=$out_dir &

#wait
#echo "concatenating temp files"
#CUDA_VISIBLE_DEVICES=7 python create_out_solo.py --split=$split --st_part=$missing_part --concat --out_dir=$out_dir
fi