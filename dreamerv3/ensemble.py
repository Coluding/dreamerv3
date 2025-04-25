import jax
import jax.numpy as jnp
from typing import Dict, List, Callable, Optional, Tuple, Literal, Union
import numpy as np
from dreamerv3 import rssm
from embodied.jax.heads import MLPHead
import elements


class EnsembleController:
    def __init__(
            self,
            ensembles: List[rssm.RSSM],
            horizon: int,
            rew_heads: Union[List[MLPHead], MLPHead],
            exploration_type: Literal["state_disagreement", "reward_variance"] = "state_disagreement",
            reward_scale: float = 1.0,
    ):
        """
        Args:
            ensembles: List of world models
            horizon: Planning horizon
            rew_heads: List of reward predictor heads (one per ensemble member)
            exploration_type: Either "state_disagreement" (original) or "reward_variance" (Plan2Explore style)
            reward_scale: Scaling factor for intrinsic rewards
        """

        if not isinstance(rew_heads, list):
            rew_heads = [rew_heads]

        if len(ensembles) != len(rew_heads):
            rew_heads = [rew_heads[0] for _ in range(len(ensembles))]

        self.models = ensembles
        self.horizon = horizon
        self.rew_heads = rew_heads
        self.exploration_type = exploration_type
        self.reward_scale = reward_scale

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

    def imagine_ensemble(
            self,
            carries: Dict,
            policy_fn: Callable,
            training: bool = True
    ) -> Tuple[List, List]:
        """Run imagination step for all ensemble members."""
        all_feats = []
        all_actions = []

        for i, model in enumerate(self.models):
            _, feat, actions = model.imagine(
                carries, policy_fn, self.horizon, training)
            all_feats.append(feat)
            all_actions.append(actions)

        return all_feats, all_actions

    def compute_disagreement(
            self,
            carries: Dict,
            policy_fn: Callable,
            training: bool = False
    ) -> jnp.ndarray:
        """
        Compute disagreement based on either state variance (original) or
        reward prediction variance (Plan2Explore style).
        """
        # Imagine trajectories for each ensemble member
        ensemble_feats, _ = self.imagine_ensemble(carries, policy_fn, training)

        if self.exploration_type == "state_disagreement":
            stacked_feats = jax.tree_map(
                lambda *x: jnp.stack(x),
                *ensemble_feats
            )

            # Combine deterministic and stochastic states
            combined_states = jnp.concatenate([
                stacked_feats['deter'],
                stacked_feats['stoch'].reshape(*stacked_feats['stoch'].shape[:-2], -1)
            ], axis=-1)

            # Compute variance across ensemble dimension
            disagreement = jnp.var(combined_states, axis=0)  # [batch, horizon, state_dim]
            disagreement = jnp.mean(disagreement, axis=-1)  # [batch, horizon]

        else:  # reward_variance
            # Plan2Explore style using reward prediction variance
            reward_preds = []

            # Get reward predictions from each ensemble member
            for feat, rew_head in zip(ensemble_feats, self.rew_heads):
                # Combine deterministic and stochastic states
                feat_tensor = jnp.concatenate([
                    feat['deter'],
                    feat['stoch'].reshape(*feat['stoch'].shape[:-2], -1)
                ], axis=-1)

                # Predict rewards using each ensemble member's reward head
                pred = rew_head(feat_tensor, 2).pred()  # Shape: [batch, horizon]
                reward_preds.append(pred)

            # Stack predictions and compute variance
            stacked_preds = jnp.stack(reward_preds)  # [ensemble, batch, horizon]
            disagreement = jnp.var(stacked_preds, axis=0)  # [batch, horizon]

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