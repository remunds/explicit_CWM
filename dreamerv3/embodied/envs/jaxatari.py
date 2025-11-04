import functools
import json

import embodied
import elements
import numpy as np
import jax
import jax.numpy as jnp
import jaxatari 
from jaxatari.wrappers import JaxatariWrapper, NormalizeObservationWrapper, AtariWrapper, PixelAndObjectCentricWrapper, PixelObsWrapper, FlattenObservationWrapper
from jaxatari.games.mods.pong_mods import RandomizedEnemyWrapper, LazyEnemyWrapper

class JAXAtari(embodied.Env):

  def __init__(self, env_name, size=(84,84), obs_mode=False, seed=None, learn_delta=True, eval=False):
    # obs_mode: "pixel", "oc", "both"
    self.obs_mode = obs_mode
    self.learn_delta = learn_delta  # whether to learn delta between oc-frames
    print("JAXAtari Env. obs_mode:", obs_mode)
    base_env = jaxatari.make(env_name)
    if eval:
      print("EVAL ENV")
      base_env = LazyEnemyWrapper(base_env)
    atari_env = AtariWrapper(base_env, frame_stack_size=2,)
    # Use frame_stack_size=2 to compute $\delta$ between oc-frames
    if obs_mode == "oc" or obs_mode == "both":
      env = PixelAndObjectCentricWrapper(atari_env, do_pixel_resize=True, grayscale=True, pixel_resize_shape=size)
    else:
      env = PixelObsWrapper(atari_env, do_pixel_resize=True, grayscale=True, pixel_resize_shape=size)
    self._env = env
    # self._norm_env = NormalizeObservationWrapper(env)

    self._done = True
    self.rng = jax.random.PRNGKey(seed)
    self.last_state = self._env.reset(self.rng)[1]
    self.prev_done = True

  @property
  def obs_space(self):
    _space = self._env.observation_space()
    # remove leading batch dimension
    if self.obs_mode == "oc":
      key = 'log/image' 
      # obs_space[0] -> img, [1] -> objects
      _img, _obj = _space
      img_shape = _img.shape[1:]
    else:
      key = 'image'
      img_shape = _space.shape[1:] if self.obs_mode == "pixel" else _space[0].shape[1:]

    obs_dict = {
        key: elements.Space(np.uint8, img_shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    if self.obs_mode == "oc" or self.obs_mode == "both":
      # add object-centric observation space
      _img, _obj = _space
      obj_shape = list(_obj.shape[1:])
      # obj_shape[-1] -= 2  # remove last 2 attributes (player/enemy scores) in pong
      if self.learn_delta:
        obj_shape[-1] *= 2  # account for delta if learning delta between oc-frames 
      obj_shape = tuple(obj_shape)
      # obs_dict['oc'] = elements.Space(np.int32, obj_shape, low=0, high=255)
      obs_dict['oc'] = elements.Space(np.int32, obj_shape)
    return obs_dict

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space().n),
        'reset': elements.Space(bool),
    }

  @functools.partial(jax.jit, static_argnums=0)
  def reset(self, rng):
    obs, state = self._env.reset(rng)
    return self._obs(obs, 0.0, {}, is_first=True, is_last=False, is_terminal=False), state

  # @functools.partial(jax.jit, static_argnums=0)
  def step(self, action):
    # couldn't jit here due to the self calls and setting it to static
    first = False

    obs, state, reward, done, info = self._env.step(self.last_state, action['action'])
    obs = (obs[0][0], obs[1][0]) if self.obs_mode == "oc" or self.obs_mode == "both" else obs[0]
    self.last_state = state
    # self.prev_done = done
    return self._obs(
        obs, reward, info,
        is_first=first,
        is_last=done,
        is_terminal=done,
    )
  
  # player x140 y96 w4 h16, enemy x16 y115 w4 h16, ball x78 y115 w2 h4, score_p0, score_e0
  # x140 dx0 y68 dy19 h16 dh0 w4 dw0, x16 dx0 y4 dy7 -6 16 0 4 0 94 4 36 4 4 0 2 0 0 0 0 0
  # -> vollkommen durcheinander wtf

  @functools.partial(jax.jit, static_argnums=0)
  def vec_step(self, action, state):
    first = False
    obs, state, reward, done, info = self._env.step(state, action)
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
    # remove batch dim
    if self.obs_mode == "oc":
      img, obj = obs
      if self.learn_delta:
        img = img[-1]
        obj_delta = obj[-1] - obj[-2]  # compute delta between current and previous oc-frame
        # obj = jnp.concat([obj[-1], obj_delta], axis=-1)  # concatenate current oc-frame and delta
        obj = jnp.stack([obj[-1], obj_delta], axis=1).reshape(-1)  # interleave current oc-frame and delta
      else:
        img = img[-1]
        obj = obj[-1]
      # obs_out['oc'] = obj[..., :-2].astype(np.int32) #removing last 2 attributes (player/enemy scores) in pong
      obs_out['oc'] = obj.astype(np.int32)
      obs_out['log/image'] = img.astype(np.uint8)
    elif self.obs_mode == "both":
      img, obj = obs
      # obs_out['oc'] = obj[..., :-2].astype(np.int32) #removing last 2 attributes (player/enemy scores) in pong
      if self.learn_delta:
        print(f"learning delta, {img.shape}, {obj.shape}")
        # jax.debug.print("obj shape: {}", obj.shape)
        img = img[-1]
        obj_delta = obj[-1] - obj[-2]  # compute delta between current and previous oc-frame
        # jax.debug.print("obj[-1][x]: {}", obj[-1][8])
        # jax.debug.print("objdelta[x]: {}", obj_delta[8])
        obj = jnp.stack([obj[-1], obj_delta], axis=1).reshape(-1)  # interleave current oc-frame and delta
        # jax.debug.print("obj shape: {}", obj.shape)
        # jax.debug.print("obj[-1][dx]: {}", obj[8*2+1])
        # With env=1:
        # px py ph pw ex ey eh ew bx by bh bw se sp
        # and interleaved with dx in between works 
        # jax.debug.print("obj_delta: {}", obj_delta)
      else:
        # remove leading batch dim
        img = img[-1]
        obj = obj[-1]
      obs_out['oc'] = obj.astype(np.int32)
      obs_out['image'] = img.astype(np.uint8)
    else:
      # remove leading batch dim
      obs_out['image'] = obs[0].astype(np.uint8)
    return obs_out
  
  @functools.partial(jax.jit, static_argnums=0)
  def render(self, state):
    return self._env.render(state)