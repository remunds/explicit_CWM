import time

import cloudpickle
import elements
import numpy as np
import portal
import jax
import jax.numpy as jnp


class Driver:

  def __init__(self, make_env_fns, parallel=True, vectorized=False, **kwargs):
    assert len(make_env_fns) >= 1
    self.parallel = parallel
    self.vectorized = vectorized
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    if parallel:
      import multiprocessing as mp
      context = mp.get_context()
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
      self.stop = context.Event()
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('act_space',))
      self.act_space = self._receive(self.pipes[0])
    elif vectorized:
      # for vectorized envs like JaxAtari
      self.envs = [make_env_fns[0]()] # only instantiate a single env 
      rng = jax.random.key(0)
      rng = jax.random.split(rng, self.length)
      self.reset_state = jax.vmap(self.envs[0].reset)(rng)[1]
      # here the shape is (4, )
      acts = jnp.zeros((self.length, *self.envs[0].act_space["action"].shape), dtype=self.envs[0].act_space["action"].dtype)
      init_obs = jax.vmap(self.envs[0].vec_step)(acts, self.reset_state)
      self.vec_prev_obs = init_obs[0]
      self.vec_last_state = init_obs[1]
      # here as well I think
      self.act_space = self.envs[0].act_space
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.act_space = self.envs[0].act_space
    self.callbacks = []
    self.acts = None
    self.carry = None
    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    if self.parallel:
      [proc.kill() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    acts = self.acts
    assert all(len(x) == self.length for x in acts.values())
    assert all((isinstance(v, np.ndarray) or isinstance(v, jnp.ndarray)) for v in acts.values())
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      obs = [self._receive(pipe) for pipe in self.pipes]
      obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
    elif self.vectorized:
      # TODO: probably not required, since jaxatari automatically resets
      # Only the is_first is actually needed I think
      def reset_if_needed(condition, idx):
        is_first, new_state = jax.lax.cond(
          condition,
          lambda: (True, jax.tree.map(lambda x: x[idx], self.reset_state)),
          lambda: (False, jax.tree.map(lambda x: x[idx], self.vec_last_state)), 
        )
        return is_first, new_state
      is_firsts, states = jax.vmap(reset_if_needed)(self.vec_prev_obs['is_last'], jnp.arange(self.length))
      # unpack acts list of dicts into jnp array
      act = jnp.array([a['action'] for a in acts])
      step_output = jax.vmap(self.envs[0].vec_step)(act, states)
      self.vec_prev_obs = step_output[0] # (dict of arrays)
      # overwrite is_first for all prev is_last ones (since they were reset)
      self.vec_prev_obs['is_first'] = is_firsts
      self.vec_last_state = step_output[1]
      # obs is already batched dict of arrays
      obs = self.vec_prev_obs  # obs[image] has shape (4, 84, 84, 1)
    else:
      obs = [env.step(act) for env, act in zip(self.envs, acts)]
      obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}

    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    assert all(len(x) == self.length for x in obs.values()), obs
    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    # self.acts = {**acts, 'reset': obs['is_last'].copy()}
    self.acts = {**acts, 'reset': np.array(obs['is_last'])}
    trans = {**obs, **acts, **outs, **logs}
    for i in range(self.length):
      trn = elements.tree.map(lambda x: x[i], trans)
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(stop, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while not stop.is_set():
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        else:
          raise ValueError(f'Invalid message {msg}')
    except ConnectionResetError:
      print('Connection to driver lost')
    except Exception as e:
      pipe.send(('error', e))
      raise
    finally:
      try:
        env.close()
      except Exception:
        pass
      pipe.close()
