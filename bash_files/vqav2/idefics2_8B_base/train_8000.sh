#!/bin/bash
#SBATCH --job-name=live_8000
#SBATCH --qos=batch-long
#SBATCH --output=log_output/log_%A_%a.out
#SBATCH --error=log_error/log_%A_%a.err
#SBATCH --nodes=1 # Number of nodes required
#SBATCH --gres=gpu:1 # Number of GPUs required
#SBATCH --gpus-per-node=1 # Number of GPU per node
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --cpus-per-task=1 # Number of CPUs per task
#SBATCH --mem=100G
#SBATCH --time 80:00:00
#SBATCH --partition=gpu-large
#SBATCH --sockets-per-node=1 # Number of sockets per node
#SBATCH --cores-per-socket=8 # Number of cores per socket
#SBATCH --exclude=a100-m-01,a100-m-02,a100-m-03


module load Anaconda3
source activate
conda activate live

cd ~/LIVE-Learnable-In-Context-Vector/

# to run LIVE
DS_SKIP_CUDA_CHECK=1 python train.py run_name="vqav2_idefics_icv"                icv_module.icv_encoder.use_sigmoid=False                icv_module.icv_encoder.alpha_init_value=0.1                data_cfg.task.datasets.max_train_size=8000                data_cfg.task.datasets.few_shot_num=32                data_cfg.bs=1                data_cfg.num_workers=1                trainer.accumulate_grad_batches=2                trainer.devices=1                 icv_module.icv_lr=1e-3                 icv_module.hard_loss_weight=0.5                 data_cfg/task/datasets=vqav2                 lmm=idefics2-8B-base                trainer.precision="16-mixed" 