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
    return {
        # use neurosymbolic input 
        'nsobs': embodied.Space(np.int32, _space.shape),
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
      ns_obs, info = self._env.reset()
      # ns_obs shape is (2, 6), make it (6, 6, 1)
      # ns_state shape is (4, 6) (6: x,y for player, ball, enemy)
      return self._obs(ns_obs, 0.0, info, is_first=True)
    ns_obs, reward, truncated, terminated, info = self._env.step(action['action'])
    return self._obs(
        ns_obs, reward, info,
        is_last=(truncated or terminated),
        is_terminal=terminated)

  def _obs(
      self, ns_obs, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        nsobs=ns_obs.astype(np.int32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
    return obs


  def render(self):
    return self._env.render()
