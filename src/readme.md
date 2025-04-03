# Install
source myenv/bin/activate
module load miniconda
<!-- conda create -n venv python=3.10 -->
conda activate /home/mila/q/qingchen.hu/textgame_579/src/myenv
conda install pytorch torchvision torchaudio -c pytorch; pip install sentencepiece

Jericho enviroment and downloads games(z-machine-games-master/jericho-game-suite): https://github.com/microsoft/jericho

Sentencepiece: pip install sentencepiece

# RUN
python train.py --output_dir 'output' --rom_path 'z-machine-games-master/jericho-game-suite/905.z5' --spm_path 'unigram_8k.model' --wandb=1 --wandb_project="579_textgame"
