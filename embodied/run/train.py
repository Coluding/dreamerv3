import collections
from functools import partial as bind

import elements
import embodied
import numpy as np
####################
# Added 23/04/2025 :) #
class OptimizedReplayBuffer:
  """Enhanced replay buffer with optimized prioritization.
  
  Uses a combined priority score based on:
  - Return values (ret)
  - Advantage magnitude (adv_mag) 
  - TD errors implied by advantage (adv)
  
  s_i = λr*Ri + λa*|A_i| + λδ*δi
  where:
  - Ri = normalized returns
  - |A_i| = magnitude of advantages
  - δi = advantage values (TD errors)
  """

  def __init__(
      self, 
      lambda_r=1.0,      # Weight for returns
      lambda_adv=0.5,    # Weight for advantage magnitude
      lambda_delta=0.3,  # Weight for TD errors (advantages)
      ):
    self.lambda_r = lambda_r
    self.lambda_adv = lambda_adv
    self.lambda_delta = lambda_delta

  def compute_score(self, metrics):
    """
    Compute priority score using available agent metrics.
    """
    returns = metrics.get('ret', 0)         # Normalized returns
    adv_mag = metrics.get('adv_mag', 0)     # Advantage magnitude
    adv = metrics.get('adv', 0)             # Advantage values (TD errors)
    
    return (self.lambda_r * returns + 
            self.lambda_adv * adv_mag +
            self.lambda_delta * adv)


####################


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  logger.close()

####################
# Added 23/04/2025 #
  def train_optimized(make_agent, make_replay, make_env, make_stream, make_logger, args):
    """Training loop with optimized replay buffer prioritization."""
    # Initialize core components
    agent = make_agent()        # Create the learning agent
    replay = make_replay()      # Create replay buffer for storing experiences
    logger = make_logger()      # Create logger for tracking metrics
    
    # Initialize optimized buffer with priority weights from args
    opt_buffer = OptimizedReplayBuffer(
        lambda_r=args.lambda_r,        # Weight for return-based priority
        lambda_adv=args.lambda_adv,    # Weight for advantage magnitude
        lambda_delta=args.lambda_delta) # Weight for TD error

    # Setup training directories and counters
    logdir = elements.Path(args.logdir)
    step = logger.step                 # Global step counter
    usage = elements.Usage(**args.usage)  # Track system resource usage
    
    # Initialize metric aggregators
    train_agg = elements.Agg()         # Aggregates training metrics
    epstats = elements.Agg()           # Aggregates episode statistics
    episodes = collections.defaultdict(elements.Agg)  # Per-episode metrics
    metrics = collections.defaultdict(list)  # Collection of various metrics
    
    # Initialize FPS trackers
    policy_fps = elements.FPS()        # Track policy execution speed
    train_fps = elements.FPS()         # Track training speed

    # Initialize timing controls
    batch_steps = args.batch_size * args.batch_length
    should_train = elements.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.LocalClock(args.log_every)
    should_save = embodied.LocalClock(args.save_every)

    # Initialize episode and step callbacks
    def per_episode(ep, mode):
      length = len(ep['reward']) - 1
      score = float(ep['reward'].astype(np.float64).sum())
      sum_abs_reward = float(np.abs(ep['reward']).astype(np.float64).sum())
      episodes[mode].add(
          {'length': length, 'score': score, 'sum_abs_reward': sum_abs_reward})
      if mode == 'train':
        logger.add(episodes[mode].result(), prefix=f'{mode}_episode')

    def per_step(tran, mode):
      step.increment()
      policy_fps.step()
      if mode == 'train':
        replay.add(tran)
      for key, value in tran.items():
        if key.startswith('log_'):
          metrics[f'{mode}_{key[4:]}'].append(value)

    driver = embodied.Driver(
        [bind(make_env, i) for i in range(args.envs)])
    driver.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
    driver.on_step(lambda tran, worker: per_step(tran, mode='train'))

    # Initialize agent and replay buffer
    random = np.random.RandomState(args.seed)
    agent.train_mode()
    train_stream = iter(agent.stream(replay, batch_steps))
    carry_train = [agent.init_train(args.batch_size)]

    # Load checkpoint if specified
    if args.from_checkpoint:
      elements.load(args.from_checkpoint, {
          'agent': bind(agent.load, regex=args.from_checkpoint_regex)})

    print('Start training loop.')
    policy = lambda *args: agent.policy(*args, mode='train')
    while step < args.steps:
      driver(policy, steps=10)

      if should_train(step):
        for _ in range(should_train(step)):
          batch = next(train_stream)
          carry_train[0], outs, mets = agent.train(carry_train[0], batch)
          
          # Compute priorities using actual agent metrics
          priorities = opt_buffer.compute_score(mets)
          
          # Update replay buffer priorities if supported
          if hasattr(replay, 'prioritize') and 'indices' in batch:
            replay.prioritize(batch['indices'], priorities)
          
          train_fps.step(batch_steps)
          if 'replay' in outs:
            replay.update(outs['replay'])
          train_agg.add(mets)

      if should_log(step):
        logger.add(metrics)
        logger.add(train_agg.result(), prefix='train')
        logger.add(episodes['train'].result(), prefix='train_episode')
        logger.add(usage.stats(), prefix='usage')
        logger.add({
            'fps/policy': policy_fps.result(),
            'fps/train': train_fps.result(),
            'steps': step,
        })
        logger.write()

      if should_save(step):
        agent.save(logdir / 'variables.pkl')
        replay.save(logdir / 'replay.pkl')

    agent.save(logdir / 'variables.pkl')
    replay.save(logdir / 'replay.pkl')
  ####################
