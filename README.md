# DreamerV3-XP: Optimizing Exploration through Uncertainty Estimation

## Introduction
DreamerV3-XP is an advanced model-based reinforcement learning (RL) agent, built upon the foundational architecture of DreamerV3. Our work specifically addresses two key areas to enhance exploration efficiency and accelerate learning in complex environments. This project extends DreamerV3 by incorporating intrinsic exploration mechanisms and optimized replay sampling strategies.

## Motivations
- **No structured exploration**: The vanilla agent is driven by acting based on the environment rewards, learning can stall in environments where rewards are rare or delayed.
- **Uniform sampling from the replay buffer**: Uniform sampling dilutes important learning signals from the most informative experiences, resulting in slower training. This motivation assumes that there are trajectories that are more informative than others.

## Project Components
- **DreamerV3 Base:**
  - **World Model:** A Recurrent State-Space Model (RSSM) that learns a compact latent representation of the environment, enabling efficient planning through imagination.
  - **Actor-Critic Framework:** Policy optimization in latent space based on trajectories imagined by the world model.

- **DreamerV3-XP Extensions:**
  1. **Prioritized Replay Buffer:**
     - Incorporates trajectory prioritization based on task return, reconstruction loss, and value prediction error.
     - Emphasizes sampling informative trajectories to enhance learning efficiency.

  2. **Intrinsic Reward (Latent Reward Disagreement):**
     - Employs an ensemble of world models to estimate uncertainty through variance in reward predictions.
     - Encourages exploration of trajectories that are uncertain yet promising.

## Methods
### World Model
The RSSM model uses deterministic $h_t$ and stochastic $z_t$ components to encode environmental dynamics. Predictions about future states, rewards, and continuation probabilities are generated from these latent representations. The model is trained via a Variational Autoencoder (VAE) approach, optimized with ELBO loss.

### Intrinsic Reward Mechanism
Intrinsic rewards are calculated from ensemble disagreement in reward predictions:

$`r^{intr}_t = \frac{1}{L} \sum_{t'=t}^{t+L} \left[\bar{r}_{t'} + \frac{1}{K}\sum_{k=1}^{K}(\hat{r}_{k,t'} - \bar{r}_{t'})^2\right]`$

Where:
- $\bar{r}_{t'}$ is the mean reward prediction.
- $\hat{r}_{k,t'}$ is the reward predicted by the $k^{th}$ ensemble member.

The final reward is a convex combination of intrinsic and extrinsic rewards:

$r^{total}_t = \lambda r^{ext}_t + (1-\lambda)r^{intr}_t$

### Dynamic Reward Weighting
We dynamically adjust $\lambda$, the weight balancing intrinsic and extrinsic rewards, using either exponential decay or a moving average of episode returns to encourage exploration when beneficial.

### Optimized Replay Buffer
Trajectory prioritization leverages three metrics:
- Task return $R_i$
- Reconstruction error $\epsilon_i$
- Critic value error $\delta_i$

The priority score $s_i$ is computed as:

$s_i = (\lambda_r + \lambda_\delta \delta_i) R_i + \lambda_\epsilon \epsilon_i$

## Results and Performance
Evaluations on selected tasks from Atari100k and DeepMind Control Visual Benchmark demonstrate that DreamerV3-XP:
- Confirms the baseline performance of DreamerV3.
- Improves learning speed, especially in sparse-reward environments.
- Achieves consistently lower dynamics and prediction errors due to prioritized replay and intrinsic reward-driven exploration.

## Related Work
DreamerV3-XP is inspired by:
- **Plan2Explore:** Leveraging ensemble disagreement for intrinsic motivation.
- **Prioritized Experience Replay (PER):** Adaptive trajectory sampling to boost learning efficiency.

## Strengths
- Robust latent dynamics modeling and efficient planning.
- Significantly enhanced exploration through uncertainty-driven intrinsic rewards.
- Improved sample efficiency and faster convergence compared to uniform replay.

## Contributions
- Proposed a novel intrinsic reward formulation utilizing ensemble disagreement to enhance exploration.
- Developed and integrated an optimized replay buffer strategy to effectively prioritize informative trajectories.
- Empirically validated improvements on challenging RL benchmarks.

## Conclusion
DreamerV3-XP provides an efficient and exploration-driven extension of DreamerV3, demonstrating significant improvements in exploration, sample efficiency, and model accuracy. This framework represents a promising step toward more versatile and effective model-based RL agents.

## Acknowledgments
This work builds upon the foundational contributions of Dreamer, Plan2Explore, and Prioritized Experience Replay (PER), among other critical studies in model-based reinforcement learning.

## References
- Hafner, D. et al. DreamerV3 (2023)
- Schaul, T. et al. Prioritized Experience Replay (2015)
- Sekar, R. et al. Plan2Explore (2020)
---
<details>
<summary>Click to toggle the section on how to run the experiments and reproduce our results</summary>

# How to use the Experimental Framework
The experimental framework is designed to serve as a single point of entry for running experiments in a well-documented and structured way - to avoid that information gets lost. It also allows to create aggregated tables for use in a paper.
<br>
<br>

## Running Experiments
The experiments/experiment_definitions.py package serves as a CLI for running experiments.
The first argument is the name of the experiment function and the following arguments can be function arguments that should be passed to the experiment functions defined in experiment_definitions.py. The name of the run config is case insensitive. The structure works as follows:
```
python experiments/experiment_definitions.py experiment_function_name --optional_function_argument value
```
For instance, to run the standard experiment from the DreamerV3 Readme page:
```
python experiments/experiment_definitions.py run_standard_dreamer --name "Test Run to check functionality" --description "Just a run with 2 seeds for testing purposes" --num_seeds 2
```
<br>

## Accessing the Results
All results are stored in `dreamerv3/artifacts/results.csv`. It contains the content of the config file, the run config (preset) and all training metrics, logged at every single step.
<br>
<br>

## Creating Tables
To create tables that are aggregated over several runs of the same experiment using different seeds, you can use the tables CLI. To create a table from the results CLI, run:
```
python experiments/tables.py
```
To include/exclude metrics from the table, modify the default argument of the `process_experiment_results` function in `tables.py`. To include experiments, add/remove the names of the experiments from the `experiment_names` default argument set. The result is printed to the commandline.
<br>

# Custom Plotting Tool 

The `custom_plot.py` script provides visualization capabilities for experiment results, supporting both score metrics and training losses.

### Basic Usage

```bash
python custom_plot.py --logdir path/to/logs/ --outdir plots/
```

### Key Features

- Automatically discovers and groups runs by method, game, and seed
- Plots individual runs and statistical aggregates (mean, median)
- Supports multiple metrics visualization (scores and various loss types)
- Auto-scales y-axis based on data range (log scale for loss metrics)

### Options

```bash
# Filter by specific methods 
python custom_plot.py --method_filter default latent_reward_disagreement

# Specify custom metrics to plot
python custom_plot.py --metrics train/loss/rew train/loss/value

# Include self-normalized statistics
python custom_plot.py --stats mean self_mean

# To disable automatic log scaling for loss metrics
python custom_plot.py --auto_log_scale False
```
</details>



