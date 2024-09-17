split=train

# CUDA_VISIBLE_DEVICES=2 python create_out.py --split=$split --part=0
# CUDA_VISIBLE_DEVICES=4 python create_out.py --split=$split --part=4
# CUDA_VISIBLE_DEVICES=2 python create_out.py --split=$split --part=2 &
# CUDA_VISIBLE_DEVICES=3 python create_out.py --split=$split --part=3 &
# CUDA_VISIBLE_DEVICES=4 python create_out.py --split=$split --part=4 &
# CUDA_VISIBLE_DEVICES=5 python create_out.py --split=$split --part=5 &
# CUDA_VISIBLE_DEVICES=6 python create_out.py --split=$split --part=6 &
# CUDA_VISIBLE_DEVICES=7 python create_out.py --split=$split --part=7 &


CUDA_VISIBLE_DEVICES=0 python create_out_solo.py --split=$split --part=0 &
CUDA_VISIBLE_DEVICES=1 python create_out_solo.py --split=$split --part=1 &
CUDA_VISIBLE_DEVICES=2 python create_out_solo.py --split=$split --part=2 &
CUDA_VISIBLE_DEVICES=3 python create_out_solo.py --split=$split --part=3 &
CUDA_VISIBLE_DEVICES=4 python create_out_solo.py --split=$split --part=4 &
CUDA_VISIBLE_DEVICES=5 python create_out_solo.py --split=$split --part=5 &
CUDA_VISIBLE_DEVICES=6 python create_out_solo.py --split=$split --part=6 &
CUDA_VISIBLE_DEVICES=7 python create_out_solo.py --split=$split --part=7 &