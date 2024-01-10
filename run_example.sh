# download pretrained models
mkdir ./cl/
wget -nc -P ./cl https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/r-50-100ep.pth.tar
wget -nc -P ./cl https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/r-50-300ep.pth.tar
wget -nc -P ./cl https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar

# install env
conda create -n mococifar100 python=3.9
conda activate mococifar100
pip install .
pip install git+https://github.com/shuqike/quinine.git
pip install opacus
pip install git+https://github.com/awslabs/fast-differential-privacy.git

# single run
python3 differentially_private/baseline_train.py --config=configs/dp/cifar100.yaml --log_dir=logs/moco3_300ep_dpft_cifar100_lr_0.05_nm_0.3_mt_0_sd_0 --project_name=dpft_cifar100 --group_name=dpft_cifar100 --run_name moco3_300ep_dp_ft_cifar100_lr_0.05_nm_0.3_mt_0_sd_0 --optimizer.args.lr=0.05 --no_wandb --seed 0 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path='cl/r-50-300ep.pth.tar'