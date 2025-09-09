import functools
import json

import embodied
import elements
import numpy as np
import jax


class JAXAtari(embodied.Env):

  def __init__(self, env_name, size=(84,84), seed=None):
    import jaxatari 
    from jaxatari.wrappers import AtariWrapper, ObjectCentricWrapper, PixelObsWrapper, LogWrapper
    self._env = PixelObsWrapper(AtariWrapper(jaxatari.make(env_name), frame_stack_size=1), do_pixel_resize=True, grayscale=True, pixel_resize_shape=size)
    self._done = True
    self.rng = jax.random.PRNGKey(seed)
    self.last_state = self._env.reset(self.rng)[1]
    self.prev_done = True

  @property
  def obs_space(self):
    _space = self._env.observation_space()
    # remove leading batch dimension
    space_shape = _space.shape[1:]
    return {
        'image': elements.Space(np.uint8, space_shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space().n),
        'reset': elements.Space(bool),
    }

  @functools.partial(jax.jit, static_argnums=0)
  def _reset(self):
    print("reset jaxatari")
    self.last_state = self._env.reset(self.rng)[1]
    self.prev_done = True

  @functools.partial(jax.jit, static_argnums=0)
  def step(self, action):
    print("step jaxatari")
    first = False
    if self.prev_done:
      first = True
    obs, state, reward, done, info = self._env.step(self.last_state, action['action'])
    self.last_state = state
    self.prev_done = done
    return self._obs(
        obs, reward, info,
        is_first=first,
        is_last=done,
        is_terminal=done,
    )

  @functools.partial(jax.jit, static_argnums=0)
  def _obs(
      self, obs, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    obs = dict(
        image=obs.astype(np.int32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
    return obs


  def render(self):
    print("Render not implemented.")
    exit()
    return self._env.render()
