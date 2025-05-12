DEFAULT_RUN_CFG = {
    "logdir": r"~/logdir/{timestamp}", 
    "configs": "atari100k",
    "run.train_ratio": 32,
    "run.duration": 120,
    "run.steps": 2000,
}

REPLAY_BUFFER_RUN_CFG = {
    "logdir": r"~/logdir/{timestamp}", 
    "configs": "atari100k",
    "run.train_ratio": 32,
    "run.duration": 120,
    "run.steps": 2000,
    "replay_context": 1,
}

REPLAY_LATENT_DISAGREEMENT_CFG = {
    "logdir": r"~/logdir/{timestamp}", 
    "configs": "atari100k",
    "run.train_ratio": 32,
    "run.duration": 120,
    "run.steps": 2000,
    "replay_context": 0,
    "agent.use_intrinsic": True,
    "agent.learn_strategy": "perturbed_starts", # ema, joint, perturbed_starts
    "agent.exploration_type": "reward_variance", # state_disagreement, reward_variance
    "agent.reward_type": "disagreement", # prediction_error, disagreement, max_disagreement
}
