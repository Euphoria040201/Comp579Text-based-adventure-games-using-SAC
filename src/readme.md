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

# References
Our experiments are based on the publicly accessible Jericho environment that provides the environment for playing all games in our experiments.

The implementation is based on the DRRN and CALM:  

1. Hausknecht, M., Ammanabrolu, P., Cˆot ́e, M.A., Yuan, X.: Interactive fiction games: A colossal adventure. In: Proceedings of the AAAI Conference on Artificial
Intelligence. pp. 7903–7910 (2020)

2. Yao, S., Rao, R., Hausknecht, M., Narasimhan, K.: Keep CALM and explore: Language models for action generation in text-based games. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. pp. 8736–8754. Association for Computational Linguistics, Online (2020)

