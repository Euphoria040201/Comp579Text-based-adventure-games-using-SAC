# Install
conda install pytorch torchvision torchaudio -c pytorch

Jericho enviroment and downloads games(z-machine-games-master/jericho-game-suite): https://github.com/microsoft/jericho

Sentencepiece: pip install sentencepiece

# RUN
python train.py --output_dir 'path' --rom_path 'path/game' --spm_path 'path'

# References
Our experiments are based on the publicly accessible Jericho environment that provides the environment for playing all games in our experiments.

The implementation is based on the DRRN and CALM:  

1. Hausknecht, M., Ammanabrolu, P., Cˆot ́e, M.A., Yuan, X.: Interactive fiction games: A colossal adventure. In: Proceedings of the AAAI Conference on Artificial
Intelligence. pp. 7903–7910 (2020)

2. Yao, S., Rao, R., Hausknecht, M., Narasimhan, K.: Keep CALM and explore: Language models for action generation in text-based games. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing. pp. 8736–8754. Association for Computational Linguistics, Online (2020)
