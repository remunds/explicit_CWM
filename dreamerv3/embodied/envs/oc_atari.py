import json

import embodied
import numpy as np


class OCAtari(embodied.Env):

  def __init__(self, env_name, logs=False, logdir=None, seed=None):
    from ocatari import OCAtari as OCAtariEnv
    self._env = OCAtariEnv(env_name=env_name, mode="ram", obs_mode="obj", seed=seed)
    self._done = True

  @property
  def obs_space(self):
    return {
        # use neurosymbolic input 
        'ns_repr': embodied.Space(np.float32, self._env._env.observation_space.shape),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_reward': embodied.Space(np.float32),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      ns_repr, info = self._env.reset()
      return self._obs(ns_repr, 0.0, info, is_first=True)
    ns_repr, reward, truncated, terminated, info = self._env.step(action['action'])
    return self._obs(
        ns_repr, reward, info,
        is_last=(truncated or terminated),
        is_terminal=info['discount'] == 0)

  def _obs(
      self, ns_repr, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        ns_repr=ns_repr,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        log_reward=np.float32(info['reward'] if info else 0.0),
    )
    return obs


  def render(self):
    return self._env.render()
