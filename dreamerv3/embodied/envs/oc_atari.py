import json

import embodied
import numpy as np


class OCAtari(embodied.Env):

  def __init__(self, env_name, size=(48, 48), logs=False, logdir=None, seed=None):
    from ocatari import OCAtari as OCAtariEnv
    self._env = OCAtariEnv(env_name=env_name, mode="ram", obs_mode="obj")#, seed=seed)
    self._done = True

  @property
  def obs_space(self):
    _space = self._env.observation_space
    # self.minires = (_space.shape[1], _space.shape[1], 1) 
    self.minires = (48, 48, 1) 
    return {
        # use neurosymbolic input 
        # 'nsrepr': embodied.Space(np.uint8, _space.shape),
        'nsrepr': embodied.Space(np.uint8, self.minires), 
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      ns_repr, info = self._env.reset()
      # ns_repr shape is (2, 6), make it (6, 6, 1)
      ns_repr = ns_repr.repeat(24, axis=0).repeat(8, axis=1)[..., np.newaxis].astype(np.uint8)
      return self._obs(ns_repr, 0.0, info, is_first=True)
    ns_repr, reward, truncated, terminated, info = self._env.step(action['action'])
    ns_repr = ns_repr.repeat(24, axis=0).repeat(8, axis=1)[..., np.newaxis].astype(np.uint8)
    return self._obs(
        ns_repr, reward, info,
        # is_last=(truncated or terminated),
        is_last=truncated,
        # is_terminal=info['discount'] == 0)
        is_terminal=terminated)

  def _obs(
      self, ns_repr, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        nsrepr=ns_repr,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
    return obs


  def render(self):
    return self._env.render()
