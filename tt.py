import embodied.envs.atari
import embodied.envs.robodesk
import numpy as np

import matplotlib.pyplot as plt

env = embodied.envs.atari.Atari("pong")
env2 = embodied.envs.robodesk.RoboDesk(repeat=1)


obs = env2.step({'action': env2.sample_valid_action(), 'reset': True})
plt.imshow(obs['image'])
plt.show()
obs = env2.step({'action': env2.sample_valid_action(), 'reset': False})
plt.imshow(obs['image'])
env2.render()
plt.show()
obs = env2.step({'action': env2.sample_valid_action(), 'reset': False})
plt.imshow(obs['image'])
print(obs)
plt.show()
