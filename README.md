# Improving RL Agents in Text-Based Games: Reward Shaping and Sampling Strategies

In this work, we present a systematic study of reward shaping and sampling strategies for training Soft Actor–Critic (SAC) agents in text‑based games. Taking inspiration from potential-based shaping, we explored the impact of two additional reward shaping mechanisms: an auxiliary reward strategy and an action-dependent 'look-back' variant that leverages critic Q-values. In addition, to address the inefficiency of uniform replay sampling in sparse‑reward environments, we evaluate two targeted alternatives: a recency‑weighted heuristic and Prioritized Experience Replay (PER) based on temporal‑difference error. Furthermore, we extend our experiments to include two additional agents: Random Ensemble Mixtur SAC (REMSAC) and the Deep Reinforcement Relevance Network (DRRN). Experiments across five Jericho games show that SAC, when combined with PER and potential-based shaping, outperforms both DRRN and REMSAC baselines, highlighting the value of structured sampling and reward design in sparse, language-driven environments. 

To run the experiment, go to the ./src folder and launch the script using the following command:

```bash
python train.py --output_dir 'output' \
    --rom_path 'z-machine-games-master/jericho-game-suite/pentari.z5' \
    --env_name 'pentari' \
    --spm_path 'unigram_8k.model' \
    --wandb=1 --wandb_project="Jericho_comp579_textgame" \
    --max_steps=5000 --memory_size=10000 \
    --agent_type=SAC --sample_strat='uniform' \
    --use_aux_reward=False --seed=3
```
