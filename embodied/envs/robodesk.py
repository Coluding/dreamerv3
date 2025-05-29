import os
import numpy as np
import embodied
import elements
import collections

if 'MUJOCO_GL' not in os.environ:
  os.environ['MUJOCO_GL'] = 'egl'
os.environ['EGL_DEVICE_ID'] = '0'

from robodesk_env.robodesk import RoboDesk as RoboDeskEnv


class RoboDesk(embodied.Env):
    def __init__(
            self, task='open_slide', reward='dense', repeat=1, size=(64, 64),
            length=500, pooling=2, aggregate='max',
            clip_reward=False, seed=None):

        assert aggregate in ('max', 'mean'), aggregate
        assert pooling >= 1, pooling

        self._env = RoboDeskEnv(
            task=task,
            reward=reward,
            action_repeat=repeat,
            image_size=size[0])

        self.repeat = repeat
        self.size = size
        self.length = length
        self.pooling = pooling
        self.aggregate = aggregate
        self.clip_reward = clip_reward
        self.rng = np.random.default_rng(seed)

        obs_space = self._env.observation_space
        shape = obs_space['image'].shape
        self.buffers = collections.deque(
            [np.zeros(shape, dtype=np.uint8) for _ in range(pooling)],
            maxlen=pooling)

        self.duration = None
        self.done = True
        self._obs_buffer = None

    @property
    def obs_space(self):
        obs = {
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
            'image': elements.Space(np.uint8, self.size + (3,)),
        }

        return obs

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.float32, (5,)),
            'reset': elements.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self.done:
            self._reset()
            self.duration = 0
            self.done = False
            return self._obs(0.0, is_first=True)

        reward = 0.0
        terminal = False
        last = False
        act = action['action']

        for _ in range(self.repeat):
            obs, r, terminated, _ = self._env.step(act)
            reward += r
            self.duration += 1
            terminal = terminal or terminated
            self._render(obs['image'])
            if self.duration >= self.length:
                last = True
                break
            if terminal:
                last = True
                break

        self.done = last
        return self._obs(reward, is_last=last, is_terminal=terminal)

    def _reset(self):
        obs = self._env.reset()

        for i in range(self.pooling):
            np.copyto(self.buffers[i], obs['image'])
        self._obs_buffer = obs['image']

    def _render(self, image):
        self.buffers.appendleft(self.buffers.pop())
        np.copyto(self.buffers[0], image)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):

        if self.clip_reward:
            reward = np.sign(reward)

        obs_out = {
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }


        if self.aggregate == 'max':
            pooled = np.amax(self.buffers, axis=0)
        else:
            pooled = np.mean(self.buffers, axis=0).astype(np.uint8)

        obs_out['image'] = pooled
        return obs_out

    def sample_valid_action(self):
        return self._env.action_space.sample()

    def render(self, **kwargs):
        return self._env.render(**kwargs)
