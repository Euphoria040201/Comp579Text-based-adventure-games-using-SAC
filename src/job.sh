#!/usr/bin/env bash
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=30GB
#SBATCH --time=60:00:00
#SBATCH --cpus-per-gpu=1
#SBATCH --output=sbatch_out/textgame1.%A.out
#SBATCH --job-name=textgame1

source myenv/bin/activate
module load miniconda
conda activate /home/mila/q/qingchen.hu/textgame_579/src/myenv
cd /home/mila/q/qingchen.hu/textgame_579/src/
python train.py --output_dir 'output' \
    --rom_path 'z-machine-games-master/jericho-game-suite/plundered.z5' \
    --spm_path 'unigram_8k.model' --wandb=1 --wandb_project="jericho-games"\
    --max_steps=5000 --memory_size=10000\
    --agent_type=SAC --sample_strat='uniform'\
    --use_aux_reward=False
    