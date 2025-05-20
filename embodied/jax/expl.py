import jax
import jax.numpy as jnp
import ninjax as nj
import dreamerv3.rssm as rssm
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
from typing import Callable
import embodied

from . import nets
from embodied.jax import Optimizer


class DisagEnsemble(nj.Module):

  def __init__(self, wm: rssm.RSSM, hidden_dim: int, classes: int, config: dict):
    self.wm = wm
    self.config = config
    self.classes = classes
    self.inputs = nets.Input(config.disag_head_inputs, dims='deter')
    self.target = nets.Input(self.config.disag_target, dims='deter')

    self.nets = [
        nets.MLP(units=hidden_dim,
                 **self.config.disag_head, winit=nets.Initializer('trunc_normal', scale=i), name=f'disag{i}')
        for i in range(self.config.ensemble_size)]

  def __call__(self, traj, return_logits: bool = True):
    preds = []
    for net in self.nets:
      inp = self.inputs(traj)
      preds.append(net(inp))

    preds = jnp.array(preds)
    logits = preds.reshape(*preds.shape[:-1], -1 , self.classes)
    if return_logits:
      return logits
    categories = jnp.argmax(logits, axis=-1)
    ohe_preds = jax.nn.one_hot(categories, logits.shape[-1])
    return ohe_preds

  def loss(self, data):
    pred = self(data)[:, :, :-1]
    tar = sg(self.target(data)[:, 1:])
    tar = tar.reshape(*tar.shape[:-1], -1, self.classes) # TODO: Check if target is OHE, I guess it is!! Then the next one makes sense
    tar_classes = jnp.argmax(tar, -1) # sample or max??
    losses = []
    for i in range(len(self)):
      cat = embodied.jax.outs.Categorical(pred[i])
      losses.append(-cat.logp(tar_classes).mean())
    return jnp.array(losses).mean()

  def __len__(self):
    return len(self.nets)

  def get_modules(self):
    return self.nets