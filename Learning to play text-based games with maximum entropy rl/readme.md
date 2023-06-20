# Install
conda install pytorch torchvision torchaudio -c pytorch

Jericho enviroment and downloads games(z-machine-games-master/jericho-game-suite): https://github.com/microsoft/jericho

Sentencepiece: pip install sentencepiece

# RUN
python train.py --output_dir 'path' --rom_path 'path/game' --spm_path 'path'

# References
Our experiments are based on the publicly accessible Jericho environment that provides the environment for playing all games in our experiments.

The implementation of the game environment and logger is based on the CALM-DRRN:  
Yao,  S.et.al: Keep  CALM  and  explore: Language  models  for  action  generation  in  text-based  games. EMNLP (2020)
