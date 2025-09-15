import functools
import json

import embodied
import elements
import numpy as np
import jax
import jaxatari 
from jaxatari.wrappers import AtariWrapper, PixelAndObjectCentricWrapper, PixelObsWrapper, FlattenObservationWrapper


class JAXAtari(embodied.Env):

  def __init__(self, env_name, size=(84,84), object_centric=False, seed=None):
    self.object_centric = object_centric
    print("JAXAtari Env. Object-centric:", object_centric)
    base_env = jaxatari.make(env_name)
    atari_env = AtariWrapper(base_env, frame_stack_size=1)
    if object_centric:
      self._env = PixelAndObjectCentricWrapper(atari_env, do_pixel_resize=True, grayscale=True, pixel_resize_shape=size)
    else:
      self._env = PixelObsWrapper(atari_env, do_pixel_resize=True, grayscale=True, pixel_resize_shape=size)
    self._done = True
    self.rng = jax.random.PRNGKey(seed)
    self.last_state = self._env.reset(self.rng)[1]
    self.prev_done = True

  @property
  def obs_space(self):
    _space = self._env.observation_space()
    # remove leading batch dimension
    if self.object_centric:
      key = 'log/image' 
      # obs_space[0] -> img, [1] -> objects
      _img, _obj = _space
      img_shape = _img.shape[1:]
      obj_shape = _obj.shape[1:]
    else:
      key = 'image'
      img_shape = _space.shape[1:]
    obs_dict = {
        key: elements.Space(np.uint8, img_shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    if self.object_centric:
      obs_dict['obs'] = elements.Space(np.int32, obj_shape, low=0, high=255)
    return obs_dict

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space().n),
        'reset': elements.Space(bool),
    }

  @functools.partial(jax.jit, static_argnums=0)
  def reset(self, rng):
    return self._env.reset(rng)

  # @functools.partial(jax.jit, static_argnums=0)
  def step(self, action):
    # couldn't jit here due to the self calls and setting it to static
    first = False
    obs, state, reward, done, info = self._env.step(self.last_state, action['action'])
    obs = (obs[0][0], obs[1][0]) if self.object_centric else obs[0]

    self.last_state = state
    # self.prev_done = done
    return self._obs(
        obs, reward, info,
        is_first=first,
        is_last=done,
        is_terminal=done,
    )

  @functools.partial(jax.jit, static_argnums=0)
  def vec_step(self, action, state):
    first = False
    obs, state, reward, done, info = self._env.step(state, action)
    # remove batch dim
    obs = (obs[0][0], obs[1][0]) if self.object_centric else obs[0]
    return self._obs(
        obs, reward, info,
        is_first=first,
        is_last=done,
        is_terminal=done,
    ), state

  @functools.partial(jax.jit, static_argnums=0)
  def _obs(
      self, obs, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    obs_out = dict(
        reward=reward.astype(np.float32),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
    if self.object_centric:
      img, obj = obs
      obs_out['obs'] = obj.astype(np.int32)
      obs_out['log/image'] = img.astype(np.uint8)
    else:
      obs_out['image'] = obs.astype(np.uint8)
    return obs_out