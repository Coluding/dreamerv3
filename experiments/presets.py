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
}

