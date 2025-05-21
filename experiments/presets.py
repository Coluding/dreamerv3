DEFAULT_RUN_CFG = {
    "logdir": r"~/logdir/default/{timestamp}",
    "run.train_ratio": 32,
    "run.log_every": 10,
    "run.save_every": 2000,
    # "run.duration": 120,
    # "run.steps": 2000,
}

REPLAY_BUFFER_RUN_CFG = {
    "logdir": r"~/logdir/optimized_replay_buffer/{timestamp}", 
    "run.train_ratio": 32,
    "run.log_every": 10,
    "run.save_every": 2000,
    # "run.duration": 120,
    # "run.steps": 2000,
    # "replay.fracs": {"uniform": 0.0, "priority": 1.0, "recency": 0.0},
    "replay.fracs.uniform": 0.0,
    "replay.fracs.priority": 1.0,
    "replay.fracs.recency": 0.0,
    # "replay.prio": {"exponent": 0.8, "maxfrac": 0.5, "initial": 1.0, "zero_on_sample": False},
    "replay.prio.exponent": 0.8,
    "replay.prio.maxfrac": 0.5,
    "replay.prio.initial": 1.0,
    "replay.prio.zero_on_sample": False,
    "replay_context": 1,
}

REPLAY_LATENT_DISAGREEMENT_CFG = {
    "logdir": r"~/logdir/latent_reward_disagreement/{timestamp}", 
    "run.train_ratio": 32,
    "run.log_every": 10,
    "run.save_every": 2000,
    # "run.duration": 0,
    # "run.steps": 4000,
    "replay_context": 0,
    "agent.use_intrinsic": True,
    "agent.intrinsic.learn_strategy": "joint_mlp", # ema, joint, perturbed_starts
    "agent.intrinsic.exploration_type": "reward_variance", # state_disagreement, reward_variance
    "agent.intrinsic.reward_type": "disagreement", # prediction_error, disagreement, max_disagreement
}


# OpenAI: Cup Catch or Reacher
# 12h for RoboDesk (they used reward heads for every task)

