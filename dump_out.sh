split=test
out_dir=/home/ggbhatt/workspace/data/custom_click_new/
org=1
missing_part=2

if [[ $org -gt 0 ]]
then
echo "generating "${split}
CUDA_VISIBLE_DEVICES=0 python create_out.py --split=$split --part=0 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=1 python create_out.py --split=$split --part=1 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=2 python create_out.py --split=$split --part=2 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=3 python create_out.py --split=$split --part=3 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=4 python create_out.py --split=$split --part=4 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=5 python create_out.py --split=$split --part=5 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=6 python create_out.py --split=$split --part=6 --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=7 python create_out.py --split=$split --part=7 --out_dir=$out_dir &
else
echo "missing part "${missing_part}
CUDA_VISIBLE_DEVICES=0 python create_out_solo.py --split=$split --part=0 --st_part=$missing_part --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=1 python create_out_solo.py --split=$split --part=1 --st_part=$missing_part --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=2 python create_out_solo.py --split=$split --part=2 --st_part=$missing_part --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=3 python create_out_solo.py --split=$split --part=3 --st_part=$missing_part --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=4 python create_out_solo.py --split=$split --part=4 --st_part=$missing_part --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=5 python create_out_solo.py --split=$split --part=5 --st_part=$missing_part --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=6 python create_out_solo.py --split=$split --part=6 --st_part=$missing_part --out_dir=$out_dir &
CUDA_VISIBLE_DEVICES=7 python create_out_solo.py --split=$split --part=7 --st_part=$missing_part --out_dir=$out_dir &

wait
echo "concatenating temp files"
CUDA_VISIBLE_DEVICES=7 python create_out_solo.py --split=$split --st_part=$missing_part --concat --out_dir=$out_dir
fi