# Install
conda install pytorch torchvision torchaudio -c pytorch

Jericho enviroment and downloads games(z-machine-games-master/jericho-game-suite): https://github.com/microsoft/jericho

Sentencepiece: pip install sentencepiece

# RUN
python train.py --output_dir 'path' --rom_path 'path/game' --spm_path 'path'
