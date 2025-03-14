# conda create -n "rank1" python=3.10
# conda init
# conda activate rank1
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install pytorch-lightning
pip install tqdm
pip install pandas
pip install datasets
pip install matplotlib
pip install transformers
pip install scikit-learn
pip install wandb
pip install -U "jax[cuda12]"
pip install flax
pip install rax
pip install mmh3