import re
from typing import Dict
import chex
import elements
import embodied.jax
import embodied.jax.nets as nn
import embodied.jax.expl as expl
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
from typing import Tuple, Union, List, Dict, Literal
import math

from . import rssm
from .ensemble import EnsembleController

f32 = jnp.float32
f16 = jnp.bfloat16
i32 = jnp.int32
sg = lambda xs, skip=False: xs if skip else jax.lax.stop_gradient(xs)
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)
prefix = lambda xs, p: {f'{p}/{k}': v for k, v in xs.items()}
concat = lambda xs, a: jax.tree.map(lambda *x: jnp.concatenate(x, a), *xs)
isimage = lambda s: s.dtype == np.uint8 and len(s.shape) == 3


class Agent(embodied.jax.Agent):

  banner = [
      r"---  ___                           __   ______ ---",
      r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
      r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
      r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
  ]

  def __init__(self, obs_space, act_space, config,):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.num_total_steps = config.num_total_steps

    exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
    enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
    dec_space = {k: v for k, v in obs_space.items() if k not in exclude}
    self.enc = {
        'simple': rssm.Encoder,
    }[config.enc.typ](enc_space, **config.enc[config.enc.typ], name='enc')
    self.dyn = {
        'rssm': rssm.RSSM,
    }[config.dyn.typ](act_space, **config.dyn[config.dyn.typ], name='dyn')
    self.dec = {
        'simple': rssm.Decoder,
    }[config.dec.typ](dec_space, **config.dec[config.dec.typ], name='dec')

    self.feat2tensor = lambda x: jnp.concatenate([
        nn.cast(x['deter'],),
        nn.cast(x['stoch'].reshape((*x['stoch'].shape[:-2], -1)))], -1)

    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    self.rew = embodied.jax.MLPHead(scalar, **config.rewhead, name='rew')
    self.con = embodied.jax.MLPHead(binary, **config.conhead, name='con')

    d1, d2 = config.policy_dist_disc, config.policy_dist_cont
    outs = {k: d1 if v.discrete else d2 for k, v in act_space.items()}
    self.pol = embodied.jax.MLPHead(
        act_space, outs, **config.policy, name='pol') # We can create with outs a DictHead output for multiple actions

    self.val = embodied.jax.MLPHead(scalar, **config.value, name='val')
    self.slowval = embodied.jax.SlowModel(
        embodied.jax.MLPHead(scalar, **config.value, name='slowval'),
        source=self.val, **config.slowvalue)

    self.retnorm = embodied.jax.Normalize(**config.retnorm, name='retnorm')
    self.valnorm = embodied.jax.Normalize(**config.valnorm, name='valnorm')
    self.advnorm = embodied.jax.Normalize(**config.advnorm, name='advnorm')
    self.extr_norm = embodied.jax.Normalize(**config.retnorm, name='extrnorm')
    self.intr_norm = embodied.jax.Normalize(**config.retnorm, name='intrnorm')

    scales = self.config.loss_scales.copy()
    rec = scales.pop('rec')
    scales.update({k: rec for k in dec_space})
    self.scales = scales

    self.modules = [
      self.dyn, self.enc, self.dec, self.rew, self.con, self.pol, self.val]
    self.opt = embodied.jax.Optimizer(
      self.modules, self._make_opt(**config.opt), summary_depth=1,
      name='opt')

    self.ensemble = None

    if config.use_intrinsic:
      print("-"*40 + "-----You are using intrinsic rewards!-----" + "-"*40)
      print("The strategy is: " + config.intrinsic["learn_strategy"])
      print("The exploration type is: " + config.intrinsic["exploration_type"])
      print("-" * 100)
      self.initialize_ensemble(act_space, enc_space, dec_space)

      self.intrinsic_reward_lambda = nj.Variable(lambda: jnp.array(1, f32), name="intrinsic_reward_lambda")
      self.intrinsic_step_counter = nj.Variable(lambda: jnp.array(0, jnp.int32), name='step_counter')
      self.extrinsic_ema = nj.Variable(lambda: jnp.array(0, f32), name="extrinsic_ema")
      self.decay_rate = nj.Variable(lambda: jnp.array(self.config.intrinsic['ema_decay'], f32), name="intrinsic_decay_rate")
      self.extrinsic_reward_ema_storage = nj.Variable(lambda: jnp.zeros(self.config.intrinsic['ema_storage_size'], f32), name="extrinsic_reward_ema_storage")
      self.extrinsic_reward_ema_storage_normed = nj.Variable(lambda: jnp.zeros(self.config.intrinsic['ema_storage_size'], f32), name="extrinsic_reward_ema_storage_normed")
      self.total_steps = nj.Variable(lambda: jnp.array(self.config.intrinsic['max_steps'] if self.config.intrinsic['max_steps'] > -1 else self.num_total_steps, jnp.int32), name="total_steps")

      self.ens_controller = EnsembleController(
        ensembles=self.ensemble if self.config.intrinsic["learn_strategy"] != "perturbed_starts" else [self.dyn],
        horizon=self.config.intrinsic["imag_horizon"],
        rew_heads=self.scaled_reward_head if self.config.intrinsic["learn_strategy"] not in ["perturbed_starts", "ema", "joint_mlp"] else self.rew,
        ensemble_method="perturbed_starts" if self.config.intrinsic["learn_strategy"] == "perturbed_starts" else "multiple_models",
        exploration_type=self.config.intrinsic["exploration_type"],
        reward_scale=self.config.intrinsic["reward_scale"],
        add_mean=self.config.intrinsic["add_mean"],
        seed=self.config.seed
      )

  @property
  def policy_keys(self):
    return '^(enc|dyn|dec|pol)/'

  @property
  def ext_space(self):
    spaces = {}
    spaces['consec'] = elements.Space(np.int32)
    spaces['stepid'] = elements.Space(np.uint8, 20)
    if self.config.replay_context:
      spaces.update(elements.tree.flatdict(dict(
          enc=self.enc.entry_space,
          dyn=self.dyn.entry_space,
          dec=self.dec.entry_space)))
    return spaces

  def init_policy(self, batch_size):
    zeros = lambda x: jnp.zeros((batch_size, *x.shape), x.dtype)
    return (
        self.enc.initial(batch_size),
        self.dyn.initial(batch_size),
        self.dec.initial(batch_size),
        jax.tree.map(zeros, self.act_space))

  def init_train(self, batch_size):
    return self.init_policy(batch_size)

  def init_report(self, batch_size):
    return self.init_policy(batch_size)

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, enc_entry, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    dec_entry = {}
    if dec_carry:
      dec_carry, dec_entry, recons = self.dec(dec_carry, feat, reset, **kw)
    policy = self.pol(self.feat2tensor(feat), bdims=1)
    act = sample(policy)
    out = {}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    if self.config.replay_context:
      out.update(elements.tree.flatdict(dict(
          enc=enc_entry, dyn=dyn_entry, dec=dec_entry)))
    return carry, act, out

  def train(self, carry: Union[List[Dict[str, jnp.ndarray]], Dict[str, jnp.ndarray]], data,): # need to go in here
    if not isinstance(carry, list):
      carry = [carry]

    combined_datarrr = [self._apply_replay_context(carry[whyeasynames], data) for whyeasynames in range(len(carry))]
    carry, obs, prevact, stepid = combined_datarrr[0]
    metrics, (carry, entries, outs, mets) = self.opt(
        self.loss, carry, obs, prevact, training=True, has_aux=True,)

    if self.config.use_intrinsic:
      if self.config.intrinsic.learn_strategy == "joint_mlp":
          policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
          data_prep = self.ens_controller.prepare_data(carry[1], policyfn)
          metrics_ens = self.ensemble_opt(
              self.ensemble.loss,
              data=data_prep
          )
          metrics.update(metrics_ens)

      elif self.config.intrinsic.learn_strategy == "joint_wm":
        metrics_ens, _ = self.ensemble_opt(
          self.ensemble.loss,
          data=carry
        )
        metrics.update(metrics_ens)

      elif self.config.intrinsic.learn_strategy == "ema":
        for model in self.ensemble:
          model.update()     

    metrics.update(mets)
    image_prio = metrics.pop('image_loss_prio')
    val_prio = metrics.pop('val_loss_prio')
    ret_prio = metrics.pop('ret_prio')
    self.slowval.update()

    outs = {}
    if self.config.replay_context:
      priorities = compute_priority(image_prio, val_prio, ret_prio)
      updates = elements.tree.flatdict(dict(
          stepid=stepid, enc=entries[0], dyn=entries[1], dec=entries[2], priority=priorities))
      B, T = obs['is_first'].shape
      assert all(x.shape[:2] == (B, T) for x in updates.values()), (
          (B, T), {k: v.shape for k, v in updates.items()})
      outs['replay'] = updates
    carry = (*carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, outs, metrics

  def world_model_loss_ensemble(self, carry: Tuple[Dict[str, jnp.ndarray]],
                                obs: Dict[str, jnp.ndarray],
                                prevact: Dict[str, jnp.ndarray],
                                training: bool, ensemble_idx: int):
    enc_carry, dyn_carry, dec_carry = carry
    reset = obs['is_first']
    B, T = reset.shape
    shapes = {}
    shapes["B"] = B
    shapes["T"] = T
    losses = {}
    metrics = {}

    # cast the carry to bfloat16
    #enc_carry = jax.tree.map(lambda x: x.astype(f16) if x.dtype == f32 else x, enc_carry)
    #dyn_carry = jax.tree.map(lambda x: x.astype(f16) if x.dtype == f32 else x, dyn_carry)
    #dec_carry = jax.tree.map(lambda x: x.astype(f16) if x.dtype == f32 else x, dec_carry)

    wm_loss, (carry, enc_entries, dec_entries, dyn_entries, tokens, repfeat) = (
        self._world_model_loss(carry, reset, losses, metrics, obs, prevact, training, ensemble_idx)
    )

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    return wm_loss, (carry, entries, outs, metrics)

  def joint_world_model_loss(self, carry, obs, prevact, training, **kw):
    """Loss aggegation"""
    tot_loss = 0.0
    all_metrics = {}
    for i in range(self.config.intrinsic.ensemble_size):
        loss_i, (_, _, _, mets_i) = self.world_model_loss_ensemble(
            ensemble_idx=i,
            carry=carry,
            obs=obs,
            prevact=prevact,
            training=training,
            **kw,
        )
        tot_loss += loss_i
        all_metrics.update(prefix(mets_i, f"ens{i}"))
    tot_loss /= self.config.intrinsic.ensemble_size  
    return tot_loss, (carry, {}, {}, all_metrics)

  def _world_model_loss(self, carry: Tuple[Dict[str, jnp.ndarray]], reset: jnp.ndarray,
                       losses: Dict[str, jnp.ndarray],
                       metrics: Dict[str, jnp.ndarray], obs: Dict[str, jnp.ndarray],
                       prevact: Dict[str, jnp.ndarray],
                       training: bool, ensemble_idx: int = -1):
    enc_carry, dyn_carry, dec_carry = carry

    enc_carry, enc_entries, tokens = self.scaled_encoder(
        enc_carry, obs, reset, training)
    dyn_model = self.ensemble[ensemble_idx] if ensemble_idx >= 0 else self.dyn
    dyn_carry, dyn_entries, los, repfeat, mets = dyn_model.loss(
        dyn_carry, tokens, prevact, reset, training)

    losses.update(los)
    metrics.update(mets)
    dec_carry, dec_entries, recons = self.scaled_decoder(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    losses['rew'] = self.scaled_reward_head(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])
    if self.config.contdisc:
        con *= 1 - 1 / self.config.horizon
    losses['con'] = self.scaled_con(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
        space, value = self.obs_space[key], obs[key]
        assert value.dtype == space.dtype, (key, space, value.dtype)
        target = f32(value) / 255 if isimage(space) else value
        losses[key] = recon.loss(sg(target))

    wm_loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
    return wm_loss, (carry, enc_entries, dec_entries, dyn_entries, tokens, repfeat)

  def loss(self, carry, obs, prevact, training, idx: int = -1,):
    enc_carry, dyn_carry, dec_carry = carry
    metrics_prefix = lambda x: f'ensemble_{idx}/{x}' if idx >= 0 else ""
    reset = obs['is_first']
    B, T = reset.shape
    losses = {}
    metrics = {}

    # World model
    enc_model = self.scaled_encoder if idx >= 0 else self.enc
    enc_carry, enc_entries, tokens = enc_model(
        enc_carry, obs, reset, training)

    dyn_model = self.ensemble[idx] if idx >= 0 else self.dyn
    dyn_carry, dyn_entries, los, repfeat, mets = dyn_model.loss(
        dyn_carry, tokens, prevact, reset, training) # dyn_carry is the last one, dyn_entries is the latent state of all items in the sequence
    losses.update(los)
    metrics.update(mets)

    dec_model = self.scaled_decoder if idx >= 0 else self.dec
    dec_carry, dec_entries, recons = dec_model(
        dec_carry, repfeat, reset, training)
    inp = sg(self.feat2tensor(repfeat), skip=self.config.reward_grad)
    reward_model = self.scaled_reward_head if idx >= 0 else self.rew
    losses['rew'] = reward_model(inp, 2).loss(obs['reward'])
    con = f32(~obs['is_terminal'])

    if self.config.contdisc:
      con *= 1 - 1 / self.config.horizon

    cont_model = self.scaled_con if idx >= 0 else self.con
    losses['con'] = cont_model(self.feat2tensor(repfeat), 2).loss(con)
    for key, recon in recons.items():
      space, value = self.obs_space[key], obs[key]
      assert value.dtype == space.dtype, (key, space, value.dtype)
      target = f32(value) / 255 if isimage(space) else value
      losses[key] = recon.loss(sg(target))
      B, T = reset.shape

    if idx >= 0:
      loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])
      return loss, (carry, metrics)

    # Imagination
    K = min(self.config.imag_last or T, T)
    H = self.config.imag_length # Imagination horizon
    starts = self.dyn.starts(dyn_entries, dyn_carry, K) # we only use the last K steps of oru observed trajectories as starting points for imagination. We flatten all of the corresponding hidden states into an array of shape B*K,D which are the starting points for the imagination

    if self.config.use_intrinsic:
        intrinsic_reward = self.compute_intrinsic_reward(starts=starts,)

    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    _, imgfeat, imgprevact = self.dyn.imagine(starts, policyfn, H, training) # imgfeat are the
    first = jax.tree.map(
        lambda x: x[:, -K:].reshape((B * K, 1, *x.shape[2:])), repfeat)
    imgfeat = concat([sg(first, skip=self.config.ac_grads), sg(imgfeat)], 1)
    lastact = policyfn(jax.tree.map(lambda x: x[:, -1], imgfeat))
    lastact = jax.tree.map(lambda x: x[:, None], lastact)
    imgact = concat([imgprevact, lastact], 1)
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgfeat))
    assert all(x.shape[:2] == (B * K, H + 1) for x in jax.tree.leaves(imgact))
    inp = self.feat2tensor(imgfeat)
    rew = self.rew(inp, 2).pred()
    
    if self.config.use_intrinsic:
      intrinsic_mets = self._intrinsic_reward_lambda_step(rew)
      metrics.update(intrinsic_mets)

    los, imgloss_out, mets = imag_loss(
        imgact,
        rew,
        self.con(inp, 2).prob(1),
        self.pol(inp, 2),
        self.val(inp, 2),
        self.slowval(inp, 2),
        self.retnorm, self.valnorm, self.advnorm, self.intr_norm, self.extr_norm,
        update=training,
        contdisc=self.config.contdisc,
        horizon=self.config.horizon,
        intrinsic_reward=intrinsic_reward if self.config.use_intrinsic else None,
        intrinsic_reward_lambda=self.intrinsic_reward_lambda.read() if self.config.use_intrinsic else None,
        **self.config.imag_loss)
    losses.update({k: v.mean(1).reshape((B, K)) for k, v in los.items()})
    metrics.update(mets)

    # Replay
    if self.config.repval_loss:
      feat = sg(repfeat, skip=self.config.repval_grad)
      last, term, rew = [obs[k] for k in ('is_last', 'is_terminal', 'reward')]
      boot = imgloss_out['ret'][:, 0].reshape(B, K)
      feat, last, term, rew, boot = jax.tree.map(
          lambda x: x[:, -K:], (feat, last, term, rew, boot))
      inp = self.feat2tensor(feat)
      los, reploss_out, mets = repl_loss(
          last, term, rew, boot,
          self.val(inp, 2),
          self.slowval(inp, 2),
          self.valnorm,
          update=training,
          horizon=self.config.horizon,
          **self.config.repl_loss)
      losses.update(los)
      metrics.update(prefix(mets, 'reploss'))

    assert set(losses.keys()) == set(self.scales.keys()), (
        sorted(losses.keys()), sorted(self.scales.keys()))
    metrics['image_loss_prio'] = sg(losses['image']).copy()
    metrics.update({f'loss/{k}': v.mean() for k, v in losses.items()})
    loss = sum([v.mean() * self.scales[k] for k, v in losses.items()])

    carry = (enc_carry, dyn_carry, dec_carry)
    entries = (enc_entries, dyn_entries, dec_entries)
    outs = {'tokens': tokens, 'repfeat': repfeat, 'losses': losses}
    # metrics.pop("intrinsic_reward") if "intrinsic_reward" in metrics else None

    return loss, (carry, entries, outs, metrics)

  def report(self, carry, data):
    if not self.config.report:
      return carry, {}

    carry, obs, prevact, _ = self._apply_replay_context(carry, data)
    (enc_carry, dyn_carry, dec_carry) = carry
    B, T = obs['is_first'].shape
    RB = min(6, B)
    metrics = {}

    # Train metrics
    _, (new_carry, entries, outs, mets) = self.loss(
        carry, obs, prevact, training=False)
    mets.update(mets)

    # Grad norms
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              carry, obs, prevact, training=False)[1][2]['losses'][key].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    # Open loop
    firsthalf = lambda xs: jax.tree.map(lambda x: x[:RB, :T // 2], xs)
    secondhalf = lambda xs: jax.tree.map(lambda x: x[:RB, T // 2:], xs)
    dyn_carry = jax.tree.map(lambda x: x[:RB], dyn_carry)
    dec_carry = jax.tree.map(lambda x: x[:RB], dec_carry)
    dyn_carry, _, obsfeat = self.dyn.observe(
        dyn_carry, firsthalf(outs['tokens']), firsthalf(prevact),
        firsthalf(obs['is_first']), training=False)
    _, imgfeat, _ = self.dyn.imagine(
        dyn_carry, secondhalf(prevact), length=T - T // 2, training=False)
    dec_carry, _, obsrecons = self.dec(
        dec_carry, obsfeat, firsthalf(obs['is_first']), training=False)
    dec_carry, _, imgrecons = self.dec(
        dec_carry, imgfeat, jnp.zeros_like(secondhalf(obs['is_first'])),
        training=False)

    # Video preds
    for key in self.dec.imgkeys:
      assert obs[key].dtype == jnp.uint8
      true = obs[key][:RB]
      pred = jnp.concatenate([obsrecons[key].pred(), imgrecons[key].pred()], 1)
      pred = jnp.clip(pred * 255, 0, 255).astype(jnp.uint8)
      error = ((i32(pred) - i32(true) + 255) / 2).astype(np.uint8)
      video = jnp.concatenate([true, pred, error], 2)

      video = jnp.pad(video, [[0, 0], [0, 0], [2, 2], [2, 2], [0, 0]])
      mask = jnp.zeros(video.shape, bool).at[:, :, 2:-2, 2:-2, :].set(True)
      border = jnp.full((T, 3), jnp.array([0, 255, 0]), jnp.uint8)
      border = border.at[T // 2:].set(jnp.array([255, 0, 0], jnp.uint8))
      video = jnp.where(mask, video, border[None, :, None, None, :])
      video = jnp.concatenate([video, 0 * video[:, :10]], 1)

      B, T, H, W, C = video.shape
      grid = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
      metrics[f'openloop/{key}'] = grid

    carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
    return carry, metrics

  def _apply_replay_context(self, carry, data):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    carry = (enc_carry, dyn_carry, dec_carry)
    stepid = data['stepid']
    obs = {k: data[k] for k in self.obs_space}
    prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
    prevact = {k: prepend(prevact[k], data[k]) for k in self.act_space}
    if not self.config.replay_context:
      return carry, obs, prevact, stepid

    K = self.config.replay_context
    nested = elements.tree.nestdict(data)
    entries = [nested.get(k, {}) for k in ('enc', 'dyn', 'dec')]
    lhs = lambda xs: jax.tree.map(lambda x: x[:, :K], xs)
    rhs = lambda xs: jax.tree.map(lambda x: x[:, K:], xs)
    rep_carry = (
        self.enc.truncate(lhs(entries[0]), enc_carry),
        self.dyn.truncate(lhs(entries[1]), dyn_carry),
        self.dec.truncate(lhs(entries[2]), dec_carry))
    rep_obs = {k: rhs(data[k]) for k in self.obs_space}
    rep_prevact = {k: data[k][:, K - 1: -1] for k in self.act_space}
    rep_stepid = rhs(stepid)

    first_chunk = (data['consec'][:, 0] == 0)
    carry, obs, prevact, stepid = jax.tree.map(
        lambda normal, replay: nn.where(first_chunk, replay, normal),
        (carry, rhs(obs), rhs(prevact), rhs(stepid)),
        (rep_carry, rep_obs, rep_prevact, rep_stepid))
    return carry, obs, prevact, stepid

  def _make_opt(
      self,
      lr: float = 4e-5,
      agc: float = 0.3,
      eps: float = 1e-20,
      beta1: float = 0.9,
      beta2: float = 0.999,
      momentum: bool = True,
      nesterov: bool = False,
      wd: float = 0.0,
      wdregex: str = r'/kernel$',
      schedule: str = 'const',
      warmup: int = 1000,
      anneal: int = 0,
  ):
    chain = []
    chain.append(embodied.jax.opt.clip_by_agc(agc))
    chain.append(embodied.jax.opt.scale_by_rms(beta2, eps))
    chain.append(embodied.jax.opt.scale_by_momentum(beta1, nesterov))
    if wd:
      assert not wdregex[0].isnumeric(), wdregex
      pattern = re.compile(wdregex)
      wdmask = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(wd, wdmask))
    assert anneal > 0 or schedule == 'const'
    if schedule == 'const':
      sched = optax.constant_schedule(lr)
    elif schedule == 'linear':
      sched = optax.linear_schedule(lr, 0.1 * lr, anneal - warmup)
    elif schedule == 'cosine':
      sched = optax.cosine_decay_schedule(lr, anneal - warmup, 0.1 * lr)
    else:
      raise NotImplementedError(schedule)
    if warmup:
      ramp = optax.linear_schedule(0.0, lr, warmup)
      sched = optax.join_schedules([ramp, sched], [warmup])
    chain.append(optax.scale_by_learning_rate(sched))
    return optax.chain(*chain)

  def _make_ensemble_rssm_config(self, config: elements.Config, model_size: float):
    ens_rssm_config = config.dyn.rssm.copy()
    relevant_keys = ["deter", "hidden", "stoch"]

    for key in relevant_keys:
      ens_rssm_config[key] = int(ens_rssm_config[key] * model_size)

    return ens_rssm_config

  def initialize_ensemble(self, act_space: dict, enc_space: dict, dec_space: dict):
    """Initializes the ensemble with scaled-down components."""

    if self.config.intrinsic["exploration_type"] == "perturbed_starts":
      return

    model_size = self.config.intrinsic["model_size"]
    ensemble_size = self.config.intrinsic["ensemble_size"]

    # Create scaled-down RSSM config
    intrinsic_rssm_config = self._make_ensemble_rssm_config(self.config, model_size)
    scalar = elements.Space(np.float32, ())
    binary = elements.Space(bool, (), 0, 2)
    #TODO: I think the model size cannot vary, otherwise the policy would not work anymore and we would also need a scaled_down policy
    match self.config.intrinsic["learn_strategy"]:
      case "joint_mlp":
          hidden_dim = self.config.dyn.rssm["stoch"] * self.config.dyn.rssm["classes"]
          self.ensemble = expl.DisagEnsemble(wm=self.dyn, hidden_dim = hidden_dim,
                                             classes=self.config.dyn.rssm["classes"],
                                             config=self.config.intrinsic,
                                             name="mlp_ensemble")

          self.ensemble_opt = embodied.jax.Optimizer(
            self.ensemble.get_modules(),
            self._make_opt(**self.config.intrinsic.expl_opt),
            summary_depth=1,
            name=f'opt_ensemble'
          )
      case "joint_wm":
        # Initialize ensemble RSSMs
        self.ensemble = [
          rssm.RSSM(
            act_space,
            **intrinsic_rssm_config,
            name=f'rssm_ensemble_{i}'
          ) for i in range(ensemble_size)
        ]

        self.scaled_encoder = rssm.Encoder(
            enc_space,
            **self._scale_config(self.config.enc.simple, model_size),
            name=f'enc_ensemble'
          )

        self.scaled_reward_head = embodied.jax.MLPHead(
          scalar,
          **self._scale_config(self.config.rewhead, model_size),
            name=f'rew_ensemble'
          )

        self.scaled_decoder = rssm.Decoder(
            dec_space,
            **self._scale_config(self.config.dec.simple, model_size),
            name=f'dec_ensemble'
          )

        self.scaled_con = embodied.jax.MLPHead(
            binary,
            **self._scale_config(self.config.conhead, model_size),
            name=f'con_ensemble'
          )
        
        ## Added 
        ensemble_modules = [
        *self.ensemble, # unpack list of ensemble models 
        self.scaled_encoder,
        self.scaled_decoder,
        self.scaled_reward_head,
        self.scaled_con
      ]
        # create a single optimizer for all ensemble members
        self.ensemble_opt = embodied.jax.Optimizer(
            ensemble_modules,
            self._make_opt(**self.config.opt),
            summary_depth=1,
            name=f'opt_ensemble'
          ) 

      case "ema":
        # TODO initialize differently --> maybe different rate?
        assert self.config.intrinsic["model_size"] == 1, "EMA only works with model_size = 1"
        self.ensemble = [
          embodied.jax.SlowModel(
            model = rssm.RSSM(
            self.act_space,
            **intrinsic_rssm_config,
            name=f'rssm_ensemble_{i}',

          ),
            source=self.dyn,
            rate=self.config.intrinsic["ema_decay"] ** (i*2), #current best solution imo
            every=self.config.intrinsic["update_every"],
          ) for i in range(self.config.intrinsic.ensemble_size)
        ]
        

  def _scale_config(self, config, scale_factor):
    """Scales the configuration values by the given scale factor."""
    scaled_config = config.copy()
    for key in ['units', 'layers', 'depth']:
      if key in scaled_config:
        scaled_config[key] = int(scaled_config[key] * scale_factor)
    return scaled_config

  def compute_intrinsic_reward(self, starts: List[Dict[str, jnp.ndarray]],):
    """Compute intrinsic reward using the ensemble controller."""
    if not self.config.use_intrinsic:
        return 0.0

    #TODO Use the imagined trajectory from self.loss() instead of running anothher imagaiantion

    # I think we would want to use the same starts for imagination as in the imagination part of the loss
    # Why? In the end, we would want to determine the LDA for state s by aggregating the intrinsic reward when starting at that state
    # TODO: Wait a minute: If we do that, we can also frame this maybe as a goal conditioned problem like in PEG?
    # We value the state in a trajectory with the intrinsic return when starting imagination in that state
    # This is effectively giving states higher rewards whose future trajectory gives us a high intrinsic reward
    # This is very simimlar to PEG (is this maybe the same lol???) where we train our policy to reach goal states from which the epxected intrinsic reward is high

    policyfn = lambda feat: sample(self.pol(self.feat2tensor(feat), 1))
    
    # Use the ensemble controller to compute intrinsic rewards
    intrinsic_reward = self.ens_controller.compute_intrinsic_reward(
        carries=starts,
        policy_fn=policyfn,
        training=False
    )
    
    return intrinsic_reward

  def _intrinsic_reward_lambda_step(self, rew,):
    metrics = {}

    count = self.intrinsic_step_counter.read()
    self.intrinsic_step_counter.write(count + 1)
    metrics["env_step"] = count

    if self.config.intrinsic['scheduling_strategy'] == "slope_ema":
      rew_mean = rew.mean()
      # --- Reward ema ---
      rew_ema = self.extrinsic_ema.read()
      decay = self.decay_rate.read()
      new_ema = jnp.array(rew_ema * decay + (1 - decay) * rew_mean, f32)
      rew_ema_storage = self.extrinsic_reward_ema_storage.read()
      # Remove the oldest value from the storage and add the new one
      self.extrinsic_reward_ema_storage.write(jnp.concatenate((rew_ema_storage[1:], jnp.expand_dims(new_ema, 0))))
      self.extrinsic_ema.write(new_ema)
      rew_ema_storage = self.extrinsic_reward_ema_storage.read()
      # Min max scaling across the values in the storage
      min_val = jnp.min(rew_ema_storage)
      max_val = jnp.max(rew_ema_storage)
      scaled_storage = (rew_ema_storage - min_val) / (max_val - min_val + 1e-8)
      self.extrinsic_reward_ema_storage_normed.write(scaled_storage)
      ema_storage = self.extrinsic_reward_ema_storage_normed.read()
      # Compute the slope of the EMA storage (has to be -1 < slope < 1)
      slope = ema_storage[-1] - ema_storage[0]
      intrinsic_reward_lambda = 1 / (1 + jnp.exp(slope * 3))
      self.intrinsic_reward_lambda.write(intrinsic_reward_lambda)
      metrics['slope'] = slope
      metrics ['rew_ema'] = new_ema
      metrics ['min_ema'] = min_val
      metrics ['max_ema'] = max_val
    elif self.config.intrinsic['scheduling_strategy'] == "exp_decay":
      intrinsic_reward_lambda = self.intrinsic_reward_lambda.read()
      total_steps = self.total_steps.read()
      decay = jnp.exp(-2.5 * (count / total_steps))
      self.intrinsic_reward_lambda.write(decay)
    elif self.config.intrinsic['scheduling_strategy'] == "fixed":
      intrinsic_reward_lambda = self.config.intrinsic['intrinsic_lambda_fixed']
      self.intrinsic_reward_lambda.write(intrinsic_reward_lambda)
    else:
      raise NotImplementedError(self.config.intrinsic['scheduling_strategy'])

    metrics ['intrinsic_reward_lambda'] = intrinsic_reward_lambda
    return metrics

def imag_loss(
    act, rew, con,
    policy, value, slowvalue,
    retnorm, valnorm, advnorm, intr_norm, extr_norm,
    update,
    contdisc=True,
    slowtar=True,
    horizon=333,
    lam=0.95,
    actent=3e-4,
    slowreg=1.0,
    intrinsic_reward=None,
    intrinsic_reward_lambda=1.0,
):
  losses = {}
  metrics = {}


  if intrinsic_reward is not None: #TODO get a feeling for the intrinsic reward scale
    intrinsic_reward_lambda =  jnp.clip(intrinsic_reward_lambda,  max=1, min=0.1) # max done in jax
    BT = rew.shape[0]
    intrinsic_reward_expanded_entry = jnp.concatenate((intrinsic_reward, jnp.zeros((BT, 1))), axis=-1)
    if intrinsic_reward_expanded_entry.shape != rew.shape:
        intrinsic_reward_expanded_entry = jnp.concatenate((jnp.zeros((BT, 1)), intrinsic_reward_expanded_entry), axis=-1)
    extrinsic_rew = rew.copy()
    rew = (1 - intrinsic_reward_lambda) * rew + intrinsic_reward_expanded_entry * intrinsic_reward_lambda

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset # normalize values -> Shape B*H, H --> Values for each imagined step
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 if contdisc else 1 - 1 / horizon
  weight = jnp.cumprod(disc * con, 1) / disc
  last = jnp.zeros_like(con)
  term = 1 - con # rew += intrinsic_reward

  if intrinsic_reward is not None: #TODO get a feeling for the intrinsic reward scale
    intrinsic_reward_lambda =  jnp.clip(intrinsic_reward_lambda,  max=1, min=0.1) # max done in jax
    BT = rew.shape[0]
    intrinsic_reward_expanded_entry = jnp.concatenate((intrinsic_reward, jnp.zeros((BT, 1))), axis=-1)
    if intrinsic_reward_expanded_entry.shape != rew.shape:
        intrinsic_reward_expanded_entry = jnp.concatenate((jnp.zeros((BT, 1)), intrinsic_reward_expanded_entry), axis=-1)
    extrinsic_rew = rew.copy()
    
    #ret = (1 - intrinsic_reward_lambda) * extrinsic_ret + intrinsic_ret * intrinsic_reward_lambda
    intr_offset, intr_rscale = intr_norm(intrinsic_reward_expanded_entry, update)
    extr_offset, extr_rscale = extr_norm(extrinsic_rew, update)
    intr_rew_norm = (intrinsic_reward_expanded_entry - intr_offset) / intr_rscale
    extr_rew_norm = (extrinsic_rew - extr_offset) / extr_rscale
    rew = (1 - intrinsic_reward_lambda) * extr_rew_norm + intrinsic_reward_lambda * intr_rew_norm

  ret = lambda_return(last, term, rew, tarval, tarval, disc, lam)
  
  roffset, rscale = retnorm(ret, update)
  adv = (ret - tarval[:, :-1]) / rscale # advantage as difference between return and value (normalized) ignoring the last step (because we have no return for the last step)
  aoffset, ascale = advnorm(adv, update)
  adv_normed = (adv - aoffset) / ascale
  logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
  ents = {k: v.entropy()[:, :-1] for k, v in policy.items()}
  policy_loss = sg(weight[:, :-1]) * -(
      logpi * sg(adv_normed) + actent * sum(ents.values()))
  losses['policy'] = policy_loss

  voffset, vscale = valnorm(ret, update) # this is important now. Here we can extract the value loss
  tar_normed = (ret - voffset) / vscale
  tar_padded = jnp.concatenate([tar_normed, 0 * tar_normed[:, -1:]], 1)
  losses['value'] = sg(weight[:, :-1]) * (
      value.loss(sg(tar_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]  # we extract the value loss here and store it


  ret_normed = (ret - roffset) / rscale
  metrics['adv'] = adv.mean()
  metrics["intrinsic_reward_lambda"] = intrinsic_reward_lambda
  metrics['adv_std'] = adv.std()
  metrics['adv_mag'] = jnp.abs(adv).mean()
  metrics['rew'] = rew.mean()
  metrics["intrinsic_reward"] = intrinsic_reward.mean() if intrinsic_reward is not None else 0
  metrics["extrinsic_reward"] = extrinsic_rew.mean() if intrinsic_reward is not None else rew.mean()
  metrics["intrinsic_reward_norm"] = intr_rew_norm.mean() if intrinsic_reward is not None else 0
  metrics["extrinsic_reward_norm"] = extr_rew_norm.mean() if intrinsic_reward is not None else ret_normed.mean()
  metrics['con'] = con.mean()
  metrics['ret'] = ret_normed.mean()
  metrics['val'] = val.mean()
  metrics['val_loss_prio'] = sg(losses['value']).copy()
  metrics["ret_prio"] = ret_normed.copy()
  metrics['tar'] = tar_normed.mean()
  metrics['weight'] = weight.mean()
  metrics['slowval'] = slowval.mean()
  metrics['ret_min'] = ret_normed.min()
  metrics['ret_max'] = ret_normed.max()
  metrics['ret_rate'] = (jnp.abs(ret_normed) >= 1.0).mean()
  metrics['extr_scale'] = extr_rscale if intrinsic_reward is not None else 0
  metrics['intr_scale'] = intr_rscale if intrinsic_reward is not None else 0
  metrics['extr_offset'] = extr_offset if intrinsic_reward is not None else 0
  metrics['intr_offset'] = intr_offset if intrinsic_reward is not None else 0
  for k in act:
    metrics[f'ent/{k}'] = ents[k].mean()
    if hasattr(policy[k], 'minent'):
      lo, hi = policy[k].minent, policy[k].maxent
      metrics[f'rand/{k}'] = (ents[k].mean() - lo) / (hi - lo)

  outs = {}
  outs['ret'] = ret
  return losses, outs, metrics


def repl_loss(
    last, term, rew, boot,
    value, slowvalue, valnorm,
    update=True,
    slowreg=1.0,
    slowtar=True,
    horizon=333,
    lam=0.95,
):
  losses = {}

  voffset, vscale = valnorm.stats()
  val = value.pred() * vscale + voffset
  slowval = slowvalue.pred() * vscale + voffset
  tarval = slowval if slowtar else val
  disc = 1 - 1 / horizon
  weight = f32(~last)
  ret = lambda_return(last, term, rew, tarval, boot, disc, lam)

  voffset, vscale = valnorm(ret, update)
  ret_normed = (ret - voffset) / vscale
  ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
  losses['repval'] = weight[:, :-1] * (
      value.loss(sg(ret_padded)) +
      slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]

  outs = {}
  outs['ret'] = ret
  metrics = {}

  return losses, outs, metrics

def compute_priority(reconstruction_losses: jnp.array,
                     value_losses: jnp.array,
                     returns: jnp.array,
                     lambda_r: float = 0.1,
                     lambda_delta: float = 0.4,
                     lambda_recon: float = 0.5) -> jnp.array:
  B, T = reconstruction_losses.shape
  BK, H = value_losses.shape

  if B*T == BK:
    priorities = ((lambda_recon * reconstruction_losses) + (lambda_r + lambda_delta *
                                                           value_losses.reshape(B,T,H).mean(axis=-1)) *
                  returns.reshape(B, T, H).sum(axis=-1))
    return priorities

  K = BK // B
  value_losses = value_losses.reshape(B, K, H)
  padding = jnp.zeros((B, T - K, H))
  value_losses = jnp.concatenate([padding, value_losses], axis=1)
  returns = jnp.concatenate([padding.copy(), returns.reshape(B, K, H)], axis=1)
  priorities = ((lambda_recon * reconstruction_losses) +
                (lambda_r + lambda_delta * value_losses.mean(axis=-1)) *
                returns.sum(axis=-1))

  return priorities


def lambda_return(last, term, rew, val, boot, disc, lam):
  chex.assert_equal_shape((last, term, rew, val, boot))
  rets = [boot[:, -1]]
  live = (1 - f32(term))[:, 1:] * disc
  cont = (1 - f32(last))[:, 1:] * lam
  interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
  for t in reversed(range(live.shape[1])):
    rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
  return jnp.stack(list(reversed(rets))[:-1], 1)

