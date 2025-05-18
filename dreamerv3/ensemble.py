import jax
import jax.numpy as jnp
from typing import Dict, List, Callable, Optional, Tuple, Literal, Union
import embodied.jax.nets as nn
from embodied.jax.expl import DisagEnsemble
from dreamerv3 import rssm
from embodied.jax.heads import MLPHead
import elements
import time
import random


class EnsembleController:
    def __init__(
            self,
            ensembles: Union[List[rssm.RSSM], DisagEnsemble],
            horizon: int,
            rew_heads: Union[List[MLPHead], MLPHead],
            exploration_type: Literal["state_disagreement", "reward_variance"] = "state_disagreement",
            ensemble_method: Literal["multiple_models", "perturbed_starts"] = "multiple_models",
            reward_scale: float = 1.0,
            noise_scale: float = 0.1,
            ensemble_size: int = 5,
            seed: int = 1,
            add_mean: bool = False
    ):
        """
        Args:
            ensembles: List of world models (or single model for perturbed_starts) or
            horizon: Planning horizon
            rew_heads: List of reward predictor heads (one per ensemble member)
            exploration_type: Either "state_disagreement" or "reward_variance"
            ensemble_method: Either "multiple_models" or "perturbed_starts"
            reward_scale: Scaling factor for intrinsic rewards
            noise_scale: Scale of noise to add when using perturbed_starts
            ensemble_size: Number of perturbed starting points to use when using perturbed_starts
        """

        if not isinstance(rew_heads, list):
            rew_heads = [rew_heads]

        self.ensemble_method = ensemble_method
        self.add_mean = add_mean
        
        if ensemble_method == "perturbed_starts":
            # For perturbed_starts, we only need one model
            assert len(ensembles) == 1, "perturbed_starts requires exactly one model"
            self.models = ensembles
            # But we'll use it multiple times with different starting points
            self.ensemble_size = ensemble_size
        else:
            # For traditional ensemble methods
            if len(ensembles) != len(rew_heads):
                rew_heads = [rew_heads[0] for _ in range(len(ensembles))]
            self.models = ensembles
            self.ensemble_size = len(ensembles)

        self.horizon = horizon
        self.rew_heads = rew_heads
        self.exploration_type = exploration_type
        self.reward_scale = reward_scale
        self.noise_scale = noise_scale
        self.seed = seed


    def observe_ensemble(
            self,
            carries: List[Dict],
            tokens: jnp.ndarray,
            actions: Dict[str, jnp.ndarray],
            resets: jnp.ndarray,
            training: bool = True
    ) -> Tuple[List, List, List]:
        """Run observation step for all ensemble members."""
        new_carries = []
        all_entries = []
        all_feats = []

        for i, model in enumerate(self.models):
            carry, entries, feat = model.observe(
                carries[i], tokens, actions, resets, training)
            new_carries.append(carry)
            all_entries.append(entries)
            all_feats.append(feat)

        return new_carries, all_entries, all_feats

    def prepare_data(self, carries: Dict, policy_fn, training: bool = True):
        _, feat, actions = self.models.wm.imagine(carries, policy_fn, self.horizon, training)
        actembed = nn.DictConcat(self.models.wm.act_space, 1)(actions)
        actembed = self.models.wm.embed_action(actembed)
        return {**feat, "action": actembed}

    def imagine_ensemble(
            self,
            carries: Dict,
            policy_fn: Callable,
            training: bool = True,
    ) -> Tuple[List, List]:
        """Run imagination step for all ensemble members or perturbed starting points."""
        all_feats = []
        all_actions = []

        if isinstance(self.models, DisagEnsemble):
            data = self.prepare_data(carries, policy_fn, training)
            ensemble_stoch = self.models(data, return_logits=False)
            ensemble_feats = {"deter": jnp.repeat(data["deter"][:, 1:, ...], len(self.models.nets)).reshape(len(self.models),
                                                                                                *data["deter"][:, 1:, ...].shape),
                              "stoch": ensemble_stoch[:, :, :-1, ...]}

            return ensemble_feats, None

        if self.ensemble_method == "perturbed_starts":
            # Create perturbed versions of the starting state
            model = self.models[0]  # We only have one model
            
            # Generate random keys for noise
            rng_key = jax.random.PRNGKey(self.seed)
            keys = jax.random.split(rng_key, self.ensemble_size)
            
            # Function to add noise to a carry
            def perturb_carry(carry, key):
                # Add noise to both deterministic and stochastic states
                noise_deter = jax.random.normal(
                    key, shape=carry['deter'].shape) * self.noise_scale
                noise_stoch = jax.random.normal(
                    key, shape=carry['stoch'].shape) * self.noise_scale
                
                # Create new carry with noise
                noisy_carry = {
                    'deter': carry['deter'] + noise_deter,
                    'stoch': carry['stoch'] + noise_stoch
                }
                return noisy_carry
            
            # Create perturbed starting points
            perturbed_carries = [perturb_carry(carries, key) for key in keys]

            # stack and flatten them
            perturbed_carries = jax.tree.map(
                lambda *x: jnp.stack(x).reshape(-1, *x[0].shape[1:]),
                *perturbed_carries
            )

            _, feat, actions = model.imagine(perturbed_carries, policy_fn, self.horizon, training)

            # unflatten
            all_feats = jax.tree.map(
                lambda x: x.reshape(self.ensemble_size, -1, *x.shape[1:]),
                feat
            )

        else:
            # Original implementation for real ensembles #TODO check if it works
            all_feats = []
            all_actions = []
            for model in self.models:
                _, feat, actions = model.imagine(carries, policy_fn, self.horizon, training)
                all_feats.append(feat)
                all_actions.append(actions)

            all_feats = jax.tree_map(lambda *x: jnp.stack(x), *all_feats)
            all_actions = jax.tree_map(lambda *x: jnp.stack(x), *all_actions)

        return all_feats, all_actions

    def compute_disagreement(
            self,
            carries: Dict,
            policy_fn: Callable,
            training: bool = False
    ) -> jnp.ndarray:
        """
        Compute disagreement based on either state variance or reward prediction variance.
        """
        # Imagine trajectories for each ensemble member or perturbed starting point

        ensemble_feats, _ = self.imagine_ensemble(carries, policy_fn, training)

        if self.exploration_type == "state_disagreement":

            # Combine deterministic and stochastic states
            combined_states = jnp.concatenate([
                ensemble_feats['deter'],
                ensemble_feats['stoch'].reshape(*ensemble_feats['stoch'].shape[:-2], -1)
            ], axis=-1)

            # Compute variance across ensemble dimension
            disagreement = jnp.std(combined_states, axis=0)  # [batch, horizon, state_dim]
            disagreement = jnp.mean(disagreement, axis=-1)  # [batch, horizon]

        else:  # reward_variance
            # Plan2Explore style using reward prediction variance

            combined_states = jnp.concatenate([
                ensemble_feats['deter'],
                ensemble_feats['stoch'].reshape(*ensemble_feats['stoch'].shape[:-2], -1)
            ], axis=-1)

            rew_head = self.rew_heads[0]

            apply_rew_head = lambda feat_tensor: rew_head(feat_tensor, 2).pred()
            reward_preds = jax.vmap(apply_rew_head)(combined_states)
            mean_reward = jnp.mean(reward_preds, axis=0)
            disagreement = jnp.std(reward_preds, axis=0)  # [batch, horizon]

            if self.add_mean:
                disagreement += mean_reward

        return disagreement

    def compute_intrinsic_reward(
            self,
            carries: Dict,
            policy_fn: Callable,
            training: bool = False
    ) -> jnp.ndarray:
        """Compute intrinsic rewards based on chosen exploration strategy."""
        disagreement = self.compute_disagreement(carries, policy_fn, training)
        return self.reward_scale * disagreement

    def initial(self, batch_size: int) -> List[Dict]:
        """Initialize carries for all ensemble members."""
        return [model.initial(batch_size) for model in self.models]