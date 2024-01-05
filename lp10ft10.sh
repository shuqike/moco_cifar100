# download pretrained models
mkdir ./cl/
wget -nc -P ./cl https://dl.fbaipublicfiles.com/moco-v3/r-50-100ep/r-50-100ep.pth.tar
wget -nc -P ./cl https://dl.fbaipublicfiles.com/moco-v3/r-50-300ep/r-50-300ep.pth.tar
wget -nc -P ./cl https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar

# lr 0.1
python differentially_private/baseline_train.py --config=configs/dp/cifar100lpfull.yaml --log_dir=logs/moco3_1000ep_dplp10ft10_cifar100_lr_0.1_nm_0.3_mt_0_sd_0 --project_name=dplpftyes_cifar100 --group_name=dplpftyes_cifar100 --run_name moco3_1000ep_dp_lp10ft10_cifar100_lr_0.1_nm_0.3_mt_0_sd_0 --optimizer.args.lr=0.1 --full_ft_epoch=10 --no_wandb --seed 0 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path=cl/r-50-1000ep.pth.tar

python differentially_private/baseline_train.py --config=configs/dp/cifar100lpfull.yaml --log_dir=logs/moco3_1000ep_dplp10ft10_cifar100_lr_0.1_nm_0.3_mt_0_sd_1 --project_name=dplpftyes_cifar100 --group_name=dplpftyes_cifar100 --run_name moco3_1000ep_dp_lp10ft10_cifar100_lr_0.1_nm_0.3_mt_0_sd_1 --optimizer.args.lr=0.1 --full_ft_epoch=10 --no_wandb --seed 1 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path=cl/r-50-1000ep.pth.tar

python differentially_private/baseline_train.py --config=configs/dp/cifar100lpfull.yaml --log_dir=logs/moco3_1000ep_dplp10ft10_cifar100_lr_0.1_nm_0.3_mt_0_sd_2 --project_name=dplpftyes_cifar100 --group_name=dplpftyes_cifar100 --run_name moco3_1000ep_dp_lp10ft10_cifar100_lr_0.1_nm_0.3_mt_0_sd_2 --optimizer.args.lr=0.1 --full_ft_epoch=10 --no_wandb --seed 2 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path=cl/r-50-1000ep.pth.tar

# lr 0.05
python differentially_private/baseline_train.py --config=configs/dp/cifar100lpfull.yaml --log_dir=logs/moco3_1000ep_dplp10ft10_cifar100_lr_0.05_nm_0.3_mt_0_sd_0 --project_name=dplpftyes_cifar100 --group_name=dplpftyes_cifar100 --run_name moco3_1000ep_dp_lp10ft10_cifar100_lr_0.05_nm_0.3_mt_0_sd_0 --optimizer.args.lr=0.05 --full_ft_epoch=10 --no_wandb --seed 0 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path=cl/r-50-1000ep.pth.tar

python differentially_private/baseline_train.py --config=configs/dp/cifar100lpfull.yaml --log_dir=logs/moco3_1000ep_dplp10ft10_cifar100_lr_0.05_nm_0.3_mt_0_sd_1 --project_name=dplpftyes_cifar100 --group_name=dplpftyes_cifar100 --run_name moco3_1000ep_dp_lp10ft10_cifar100_lr_0.05_nm_0.3_mt_0_sd_1 --optimizer.args.lr=0.05 --full_ft_epoch=10 --no_wandb --seed 1 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path=cl/r-50-1000ep.pth.tar

python differentially_private/baseline_train.py --config=configs/dp/cifar100lpfull.yaml --log_dir=logs/moco3_1000ep_dplp10ft10_cifar100_lr_0.05_nm_0.3_mt_0_sd_2 --project_name=dplpftyes_cifar100 --group_name=dplpftyes_cifar100 --run_name moco3_1000ep_dp_lp10ft10_cifar100_lr_0.05_nm_0.3_mt_0_sd_2 --optimizer.args.lr=0.05 --full_ft_epoch=10 --no_wandb --seed 2 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path=cl/r-50-1000ep.pth.tar

# lr 0.025
python differentially_private/baseline_train.py --config=configs/dp/cifar100lpfull.yaml --log_dir=logs/moco3_1000ep_dplp10ft10_cifar100_lr_0.025_nm_0.3_mt_0_sd_0 --project_name=dplpftyes_cifar100 --group_name=dplpftyes_cifar100 --run_name moco3_1000ep_dp_lp10ft10_cifar100_lr_0.025_nm_0.3_mt_0_sd_0 --optimizer.args.lr=0.025 --full_ft_epoch=10 --no_wandb --seed 0 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path=cl/r-50-1000ep.pth.tar

python differentially_private/baseline_train.py --config=configs/dp/cifar100lpfull.yaml --log_dir=logs/moco3_1000ep_dplp10ft10_cifar100_lr_0.025_nm_0.3_mt_0_sd_1 --project_name=dplpftyes_cifar100 --group_name=dplpftyes_cifar100 --run_name moco3_1000ep_dp_lp10ft10_cifar100_lr_0.025_nm_0.3_mt_0_sd_1 --optimizer.args.lr=0.025 --full_ft_epoch=10 --no_wandb --seed 1 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path=cl/r-50-1000ep.pth.tar

python differentially_private/baseline_train.py --config=configs/dp/cifar100lpfull.yaml --log_dir=logs/moco3_1000ep_dplp10ft10_cifar100_lr_0.025_nm_0.3_mt_0_sd_2 --project_name=dplpftyes_cifar100 --group_name=dplpftyes_cifar100 --run_name moco3_1000ep_dp_lp10ft10_cifar100_lr_0.025_nm_0.3_mt_0_sd_2 --optimizer.args.lr=0.025 --full_ft_epoch=10 --no_wandb --seed 2 --privacy_engine.args.noise_multiplier=0.3 --model.args.checkpoint_path=cl/r-50-1000ep.pth.tar