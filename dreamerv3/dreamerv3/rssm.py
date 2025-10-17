import math

import einops
import elements
import embodied.jax
import embodied.jax.nets as nn
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

f32 = jnp.float32
sg = jax.lax.stop_gradient


class RSSM(nj.Module):

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0
  oc_tokens: int = 0 # defines how many tokens in front correspond to object-centric input. If None, no OC input.

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    self.act_space = act_space
    self.kw = kw

  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action)

    if self.oc_tokens != 0:
      # additionally use object-centric input as posterior 
      oc = tokens[..., :self.oc_tokens]
      # oc_logit = nn.cast(jax.nn.one_hot(oc, self.classes, axis=-1)) # shape (..., oc_tokens, classes), pong: (1, 14, 256)
      # img_tokens = tokens[..., self.oc_tokens:].reshape((*deter.shape[:-1], -1))
      # x = img_tokens if self.absolute else jnp.concatenate([deter, img_tokens], -1)
      # for i in range(self.obslayers):
      #   x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      #   x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
      # img_logit = self._logit('obslogit', x, additional_to_oc=True) #shape (..., n_img_dim, classes), pong: (1, 2, 256)
      # logit = jnp.concatenate([oc_logit, img_logit], axis=-2) # concat along object/attribute dim, not n_values
      # stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

      std_dev = jnp.ones_like(oc) * 0.01
      logit = jnp.concat([oc, std_dev], axis=-1)
      stoch = nn.cast(self._norm_dist(logit).sample(seed=nj.seed()))

    else:
      tokens = tokens.reshape((*deter.shape[:-1], -1))
      x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
      for i in range(self.obslayers):
        x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
      logit = self._logit('obslogit', x)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    # add classes dim back 
    stoch = jnp.expand_dims(stoch, axis=-1)

    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      deter = self._core(carry['deter'], carry['stoch'], actemb)
      logit = self._prior(deter) 
      stoch = nn.cast(self._norm_dist(logit).sample(seed=nj.seed()))
      # add classes dim back
      stoch = jnp.expand_dims(stoch, axis=-1)
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)

      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._norm_dist(sg(post)).kl(self._norm_dist(prior))
    rep = self._norm_dist(post).kl(self._norm_dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._norm_dist(prior).entropy().mean()
    metrics['rep_ent'] = self._norm_dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics

  def _core(self, deter, stoch, action):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub('dynin0', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub('dynin0norm', nn.Norm, self.norm)(x0))
    x1 = self.sub('dynin1', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub('dynin1norm', nn.Norm, self.norm)(x1))
    x2 = self.sub('dynin2', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub('dynin2norm', nn.Norm, self.norm)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      x = self.sub(f'dynhid{i}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm', nn.Norm, self.norm)(x))
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter

  def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x, additional_to_oc=False):
    kw = dict(**self.kw, outscale=self.outscale)
    out_dim = self.stoch
    # if additional_to_oc:
    #   out_dim = self.stoch - self.oc_tokens
    # x = self.sub(name, nn.Linear, out_dim*self.classes, **kw)(x)
    # return x.reshape(x.shape[:-1] + (out_dim, self.classes))
    x = self.sub(name, nn.Linear, out_dim*2, **kw)(x)
    return x.reshape(x.shape[:-1] + (out_dim*2,))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out

  def _norm_dist(self, logit):
    assert self.oc_tokens == self.stoch
    mean = logit[..., :self.oc_tokens]
    std_dev = logit[..., self.oc_tokens:]
    std_dev = jax.nn.softplus(std_dev) + 1e-4  # ensure stddev is positive and not too close to zero
    # if std_dev.shape[-1] == 0:
    #   std_dev = jnp.ones_like(mean) * 0.1
    assert mean.shape == std_dev.shape
    out = embodied.jax.outs.Normal(mean, stddev=std_dev)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out

class ElementwiseRSSM(nj.Module):
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0
  oc_tokens: int = 0 # defines how many tokens in front correspond to object-centric input. If None, no OC input.
  attributes_per_object: int = 4
  n_objects: int = 3
  obj_type_mapping = lambda self, obj: str(obj)  # in pong: 0:ball, 1:player, 2:enemy

  def __init__(self, act_space, **kw):
    assert self.deter % self.blocks == 0
    assert self.n_objects * self.attributes_per_object == self.stoch, "Number of objects times attributes per object must equal stoch dimension"
    self.act_space = act_space
    self.kw = kw
    self.obj_deter = self.deter // self.n_objects 
    assert self.obj_deter * self.n_objects == self.deter, "deter must be perfectly divisible by n_objects currently"
    # TODO: What if deter is not perfectly divisible by n_objects?
  
  @property
  def entry_space(self):
    return dict(
        deter=elements.Space(np.float32, self.deter),
        stoch=elements.Space(np.float32, (self.stoch, self.classes)))

  def initial(self, bsize):
    carry = nn.cast(dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32)))
    return carry

  def truncate(self, entries, carry=None):
    assert entries['deter'].ndim == 3, entries['deter'].shape
    carry = jax.tree.map(lambda x: x[:, -1], entries)
    return carry

  def starts(self, entries, carry, nlast):
    B = len(jax.tree.leaves(carry)[0])
    return jax.tree.map(
        lambda x: x[:, -nlast:].reshape((B * nlast, *x.shape[2:])), entries)

  def observe(self, carry, tokens, action, reset, training, single=False):
    carry, tokens, action = nn.cast((carry, tokens, action))
    if single:
      carry, (entry, feat) = self._observe(
          carry, tokens, action, reset, training)
      return carry, entry, feat
    else:
      unroll = jax.tree.leaves(tokens)[0].shape[1] if self.unroll else 1
      carry, (entries, feat) = nj.scan(
          lambda carry, inputs: self._observe(
              carry, *inputs, training),
          carry, (tokens, action, reset), unroll=unroll, axis=1)
      return carry, entries, feat

  def imagine(self, carry, policy, length, training, single=False):
    if single:
      action = policy(sg(carry)) if callable(policy) else policy
      actemb = nn.DictConcat(self.act_space, 1)(action)
      update = jnp.zeros_like(carry['deter'])
      cand = jnp.zeros_like(carry['deter'])
      for obj in range(self.n_objects):
        # select all attributes for this object from stoch
        # obj0: 0-3, obj1: 4-7, obj2: 8-11, obj3 == rest: 12-15
        # obj_attribute = slice(obj * self.attributes_per_object, (obj + 1) * self.attributes_per_object)
        obj_deter_attribute = slice(obj * self.obj_deter, (obj + 1) * self.obj_deter)
        obj_type = self.obj_type_mapping(obj)
        # single_update, single_cand = self._core(obj_type, carry['deter'], carry['stoch'][:, obj_attribute], actemb) #axis 2 is obj_attribute dim
        # Testing if full stoch works better than only own obj attributes
        # single_update, single_cand = self._core(obj_type, carry['deter'][:, obj_deter_attribute], carry['stoch'], actemb) #axis 2 is obj_attribute dim
        #TODO: Currently testing whether full access to deter and carry works.
        single_update, single_cand = self._core(obj_type, carry['deter'], carry['stoch'], actemb) #axis 2 is obj_attribute dim
        start_indices = (0, obj * self.obj_deter)
        # update = jax.lax.dynamic_update_slice(update, single_update, start_indices)
        update = update.at[:, obj_deter_attribute].set(single_update)
        # cand = jax.lax.dynamic_update_slice(cand, single_cand, start_indices)
        cand = cand.at[:, obj_deter_attribute].set(single_cand)

      # TODO: Should we handle non-padded inputs?

      deter = update * cand + (1 - update) * carry['deter']

      logit = self._prior(deter)
      stoch = nn.cast(self._norm_dist(logit).sample(seed=nj.seed()))
      stoch = jnp.expand_dims(stoch, axis=-1) # add classes dim back
      carry = nn.cast(dict(deter=deter, stoch=stoch))
      feat = nn.cast(dict(deter=deter, stoch=stoch, logit=logit))
      assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
      return carry, (feat, action)
    else:
      unroll = length if self.unroll else 1
      if callable(policy):
        carry, (feat, action) = nj.scan(
            lambda c, _: self.imagine(c, policy, 1, training, single=True),
            nn.cast(carry), (), length, unroll=unroll, axis=1)
      else:
        carry, (feat, action) = nj.scan(
            lambda c, a: self.imagine(c, a, 1, training, single=True),
            nn.cast(carry), nn.cast(policy), length, unroll=unroll, axis=1)
      # We can also return all carry entries but it might be expensive.
      # entries = dict(deter=feat['deter'], stoch=feat['stoch'])
      # return carry, entries, feat, action
      return carry, feat, action

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)

    update = jnp.zeros_like(deter)
    cand = jnp.zeros_like(deter)

    for obj in range(self.n_objects):
      # select all attributes for this object from stoch
      # obj0: 0-3, obj1: 4-7, obj2: 8-11, obj3 == rest: 12-15
      # obj_attribute = slice(obj * self.attributes_per_object, (obj + 1) * self.attributes_per_object)
      obj_deter_attribute = slice(obj * self.obj_deter, (obj + 1) * self.obj_deter)
      obj_type = self.obj_type_mapping(obj)
      # print("observe: ", deter[:, obj_deter_attribute])
      # single_update, single_cand = self._core(obj_type, deter[:, obj_deter_attribute], stoch, action) #axis 2 is obj_attribute dim
      #TODO: Currently testing whether full access to deter and stoch works.
      # Did a big mistake and not updating the variables before!
      # So if current run works -> Test again with reduced deter / stoch.
      single_update, single_cand = self._core(obj_type, deter, stoch, action) #axis 2 is obj_attribute dim
      update = update.at[:, obj_deter_attribute].set(single_update)
      # start_indices = (0, obj * self.obj_deter)
      # update = jax.lax.dynamic_update_slice(update, single_update, start_indices)
      cand = cand.at[:, obj_deter_attribute].set(single_cand)
      # cand = jax.lax.dynamic_update_slice(cand, single_cand, start_indices) 

    # TODO: Should we handle non-padded inputs?

    deter = update * cand + (1 - update) * deter

    if self.oc_tokens != 0:
      # additionally use object-centric input as posterior 
      oc = tokens[..., :self.oc_tokens]
      # oc_logit = nn.cast(jax.nn.one_hot(oc, self.classes, axis=-1)) # shape (..., oc_tokens, classes), pong: (1, 14, 256)
      # img_tokens = tokens[..., self.oc_tokens:].reshape((*deter.shape[:-1], -1))
      # x = img_tokens if self.absolute else jnp.concatenate([deter, img_tokens], -1)
      # for i in range(self.obslayers):
      #   x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      #   x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
      # img_logit = self._logit('obslogit', x, additional_to_oc=True) #shape (..., n_img_dim, classes), pong: (1, 2, 256)
      # logit = jnp.concatenate([oc_logit, img_logit], axis=-2) # concat along object/attribute dim, not n_values
      # stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      std_dev = jnp.ones_like(oc) * 0.01
      logit = jnp.concat([oc, std_dev], axis=-1) 
      stoch = nn.cast(self._norm_dist(logit).sample(seed=nj.seed()))

    else:
      tokens = tokens.reshape((*deter.shape[:-1], -1))
      x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
      for i in range(self.obslayers):
        x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
      logit = self._logit('obslogit', x)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))

    stoch = jnp.expand_dims(stoch, axis=-1) # add classes dim back

    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def _core(self, obj_type, deter, stoch, action):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.sub(f'dynin0{obj_type}', nn.Linear, self.hidden, **self.kw)(deter)
    x0 = nn.act(self.act)(self.sub(f'dynin0norm{obj_type}', nn.Norm, self.norm)(x0))
    x1 = self.sub(f'dynin1{obj_type}', nn.Linear, self.hidden, **self.kw)(stoch)
    x1 = nn.act(self.act)(self.sub(f'dynin1norm{obj_type}', nn.Norm, self.norm)(x1))
    x2 = self.sub(f'dynin2{obj_type}', nn.Linear, self.hidden, **self.kw)(action)
    x2 = nn.act(self.act)(self.sub(f'dynin2norm{obj_type}', nn.Norm, self.norm)(x2))
    x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
    x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
    for i in range(self.dynlayers):
      # x = self.sub(f'dynhid{i}{obj_type}', nn.BlockLinear, self.deter, g, **self.kw)(x)
      x = self.sub(f'dynhid{i}{obj_type}', nn.BlockLinear, self.obj_deter, g, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'dynhid{i}norm{obj_type}', nn.Norm, self.norm)(x))
    # x = self.sub(f'dyngru{obj_type}', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)
    x = self.sub(f'dyngru{obj_type}', nn.BlockLinear, 3 * self.obj_deter, g, **self.kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    # deter = update * cand + (1 - update) * deter
    return update, cand
  
  def loss(self, carry, tokens, acts, reset, training):
    metrics = {}
    carry, entries, feat = self.observe(carry, tokens, acts, reset, training)
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._norm_dist(sg(post)).kl(self._norm_dist(prior))
    rep = self._norm_dist(post).kl(self._norm_dist(sg(prior)))
    if self.free_nats:
      dyn = jnp.maximum(dyn, self.free_nats)
      rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
    metrics['dyn_ent'] = self._norm_dist(prior).entropy().mean()
    metrics['rep_ent'] = self._norm_dist(post).entropy().mean()
    return carry, entries, losses, feat, metrics

  def _prior(self, feat):
    x = feat
    for i in range(self.imglayers):
      x = self.sub(f'prior{i}', nn.Linear, self.hidden, **self.kw)(x)
      x = nn.act(self.act)(self.sub(f'prior{i}norm', nn.Norm, self.norm)(x))
    return self._logit('priorlogit', x)

  def _logit(self, name, x, additional_to_oc=False):
    kw = dict(**self.kw, outscale=self.outscale)
    out_dim = self.oc_tokens if self.oc_tokens != 0 else self.stoch
    # if additional_to_oc:
    #   out_dim = self.stoch - self.oc_tokens
    # x = self.sub(name, nn.Linear, out_dim*self.classes, **kw)(x)
    # return x.reshape(x.shape[:-1] + (out_dim, self.classes))
    x = self.sub(name, nn.Linear, out_dim*2, **kw)(x)
    return x.reshape(x.shape[:-1] + (out_dim*2,))

  def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out

  def _norm_dist(self, logit):
    # assert self.oc_tokens == self.stoch
    mean = logit[..., :self.oc_tokens]
    std_dev = logit[..., self.oc_tokens:]
    if self.stoch > self.oc_tokens:
      # pad with 0s
      mean = jnp.concat([mean, jnp.zeros(mean.shape[:-1] + (self.stoch - self.oc_tokens,))], axis=-1)
      std_dev = jnp.concat([std_dev, jnp.ones(std_dev.shape[:-1] + (self.stoch - self.oc_tokens,)) * 0.01], axis=-1)
    std_dev = jax.nn.softplus(std_dev) + 1e-4  # ensure stddev is positive and not too close to zero
    # if std_dev.shape[-1] == 0:
    #   std_dev = jnp.ones_like(mean) * 0.1
    assert mean.shape == std_dev.shape
    out = embodied.jax.outs.Normal(mean, stddev=std_dev)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out

class DeltaRSSM(RSSM):
  """
  Goal: Only model changes to the state, not the entire state.
  We therefore make deter the delta state, and add it to the previous stoch state
  Kind of like a residual connection.
  """
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  blocks: int = 8
  free_nats: float = 1.0
  oc_tokens: int = 0 # defines how many tokens in front correspond to object-centric input. If None, no OC input.

  def __init__(self, act_space, **kw):
    super().__init__(act_space, **kw)
  
  def _dist(self, logits):
    return self._norm_dist(logits)

  def _observe(self, carry, tokens, action, reset, training):
    deter, stoch, action = nn.mask(
        (carry['deter'], carry['stoch'], action), ~reset)
    action = nn.DictConcat(self.act_space, 1)(action)
    action = nn.mask(action, ~reset)
    deter = self._core(deter, stoch, action)

    if self.oc_tokens != 0:
      # additionally use object-centric input as posterior 
      oc_tok = tokens[..., :self.oc_tokens]
      # split into oc and delta_oc
      # select every second token as delta
      oc = oc_tok[..., ::2]
      oc_delta = oc_tok[..., 1::2]
      #TODO: Not sure where the extra dim is coming from
      # Probably from RSSM.imagine where we add an extra dim to stoch (unnecessarily)
      new_oc = stoch.squeeze() + oc_delta 
      std_dev = jnp.ones_like(new_oc) * 0.01
      logit = jnp.concat([new_oc, std_dev], axis=-1) 
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      stoch = jnp.expand_dims(stoch, axis=-1)
      # img_tokens = tokens[..., self.oc_tokens:].reshape((*deter.shape[:-1], -1))
      # x = img_tokens if self.absolute else jnp.concatenate([deter, img_tokens], -1)
      # for i in range(self.obslayers):
      #   x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
      #   x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
      # img_logit = self._logit('obslogit', x, additional_to_oc=True) #shape (..., n_img_dim, classes), pong: (1, 2, 256)
      # logit = jnp.concatenate([oc_logit, img_logit], axis=-2) # concat along object/attribute dim, not n_values
      # stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
      # Stoch represents delta to previous oc here.
      # So prev_oc + stoch = oc 

    else:
      tokens = tokens.reshape((*deter.shape[:-1], -1))
      x = tokens if self.absolute else jnp.concatenate([deter, tokens], -1)
      for i in range(self.obslayers):
        x = self.sub(f'obs{i}', nn.Linear, self.hidden, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'obs{i}norm', nn.Norm, self.norm)(x))
      logit = self._logit('obslogit', x)
      stoch = nn.cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    feat = dict(deter=deter, stoch=stoch, logit=logit)
    entry = dict(deter=deter, stoch=stoch)
    assert all(x.dtype == nn.COMPUTE_DTYPE for x in (deter, stoch, logit))
    return carry, (entry, feat)

  def _norm_dist(self, logit):
    mean = logit[..., :self.stoch]
    std_dev = logit[..., self.stoch:]
    std_dev = jax.nn.softplus(std_dev) + 1e-4  # ensure stddev is positive and not too close to zero
    assert mean.shape == std_dev.shape
    out = embodied.jax.outs.Normal(mean, stddev=std_dev)
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out
  

class DummyEncoder:#(nj.Module):
  """
  Pass inputs as-is, without any changes. This is useful for object-centric inputs.
  """

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    x = obs["oc"]
    tokens = x
    entries = {}
    return carry, entries, tokens
  
class Encoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  outer: bool = False
  strided: bool = False
  oc_passthrough: bool = True  # If True, pass object-centric obs as-is.

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, obs, reset, training, single=False):
    bdims = 1 if single else 2
    outs = []
    bshape = reset.shape

    if self.oc_passthrough and 'oc' in self.veckeys:
      x = obs['oc']
      x = x.reshape((-1, *x.shape[bdims:]))
      outs.append(x)

    elif self.veckeys:
      vspace = {k: self.obs_space[k] for k in self.veckeys}
      vecs = {k: obs[k] for k in self.veckeys}
      squish = nn.symlog if self.symlog else lambda x: x
      x = nn.DictConcat(vspace, 1, squish=squish)(vecs)
      x = x.reshape((-1, *x.shape[bdims:]))
      for i in range(self.layers):
        x = self.sub(f'mlp{i}', nn.Linear, self.units, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'mlp{i}norm', nn.Norm, self.norm)(x))
      outs.append(x)

    if self.imgkeys:
      K = self.kernel
      imgs = [obs[k] for k in sorted(self.imgkeys)]
      assert all(x.dtype == jnp.uint8 for x in imgs)
      x = nn.cast(jnp.concatenate(imgs, -1), force=True) / 255 - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        if self.outer and i == 0:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
        elif self.strided:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, 2, **self.kw)(x)
        else:
          x = self.sub(f'cnn{i}', nn.Conv2D, depth, K, **self.kw)(x)
          B, H, W, C = x.shape
          x = x.reshape((B, H // 2, 2, W // 2, 2, C)).max((2, 4))
        x = nn.act(self.act)(self.sub(f'cnn{i}norm', nn.Norm, self.norm)(x))
      assert 3 <= x.shape[-3] <= 16, x.shape
      assert 3 <= x.shape[-2] <= 16, x.shape
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    tokens = x.reshape((*bshape, *x.shape[1:]))
    entries = {}
    return carry, entries, tokens


class Decoder(nj.Module):

  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  depth: int = 64
  mults: tuple = (2, 3, 4, 4)
  layers: int = 3
  kernel: int = 5
  symlog: bool = True
  bspace: int = 8
  outer: bool = False
  strided: bool = False

  def __init__(self, obs_space, **kw):
    assert all(len(s.shape) <= 3 for s in obs_space.values()), obs_space
    self.obs_space = obs_space
    self.veckeys = [k for k, s in obs_space.items() if len(s.shape) <= 2]
    if "oc" in self.veckeys:
      self.veckeys.remove("oc") 
    self.imgkeys = [k for k, s in obs_space.items() if len(s.shape) == 3]
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.imgdep = sum(obs_space[k].shape[-1] for k in self.imgkeys)
    self.imgres = self.imgkeys and obs_space[self.imgkeys[0]].shape[:-1]
    self.kw = kw

  @property
  def entry_space(self):
    return {}

  def initial(self, batch_size):
    return {}

  def truncate(self, entries, carry=None):
    return {}

  def __call__(self, carry, feat, reset, training, single=False):
    assert feat['deter'].shape[-1] % self.bspace == 0
    K = self.kernel
    recons = {}
    bshape = reset.shape
    inp = [nn.cast(feat[k]) for k in ('stoch', 'deter')]
    inp = [x.reshape((math.prod(bshape), -1)) for x in inp]
    inp = jnp.concatenate(inp, -1)

    if self.veckeys:
      spaces = {k: self.obs_space[k] for k in self.veckeys}
      o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
      outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}
      kw = dict(**self.kw, act=self.act, norm=self.norm)
      x = self.sub('mlp', nn.MLP, self.layers, self.units, **kw)(inp)
      x = x.reshape((*bshape, *x.shape[1:]))
      kw = dict(**self.kw, outscale=self.outscale)
      outs = self.sub('vec', embodied.jax.DictHead, spaces, outputs, **kw)(x)
      recons.update(outs)

    if self.imgkeys:
      factor = 2 ** (len(self.depths) - int(bool(self.outer)))
      minres = [int(x // factor) for x in self.imgres]
      assert 3 <= minres[0] <= 16, minres
      assert 3 <= minres[1] <= 16, minres
      shape = (*minres, self.depths[-1])
      if self.bspace:
        u, g = math.prod(shape), self.bspace
        x0, x1 = nn.cast((feat['deter'], feat['stoch']))
        x1 = x1.reshape((*x1.shape[:-2], -1))
        x0 = x0.reshape((-1, x0.shape[-1]))
        x1 = x1.reshape((-1, x1.shape[-1]))
        x0 = self.sub('sp0', nn.BlockLinear, u, g, **self.kw)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=minres[0], w=minres[1], g=g)
        x1 = self.sub('sp1', nn.Linear, 2 * self.units, **self.kw)(x1)
        x1 = nn.act(self.act)(self.sub('sp1norm', nn.Norm, self.norm)(x1))
        x1 = self.sub('sp2', nn.Linear, shape, **self.kw)(x1)
        x = nn.act(self.act)(self.sub('spnorm', nn.Norm, self.norm)(x0 + x1))
      else:
        x = self.sub('space', nn.Linear, shape, **kw)(inp)
        x = nn.act(self.act)(self.sub('spacenorm', nn.Norm, self.norm)(x))
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        if self.strided:
          kw = dict(**self.kw, transp=True)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, 2, **kw)(x)
        else:
          x = x.repeat(2, -2).repeat(2, -3)
          x = self.sub(f'conv{i}', nn.Conv2D, depth, K, **self.kw)(x)
        x = nn.act(self.act)(self.sub(f'conv{i}norm', nn.Norm, self.norm)(x))
      if self.outer:
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      elif self.strided:
        kw = dict(**self.kw, outscale=self.outscale, transp=True)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, 2, **kw)(x)
      else:
        x = x.repeat(2, -2).repeat(2, -3)
        kw = dict(**self.kw, outscale=self.outscale)
        x = self.sub('imgout', nn.Conv2D, self.imgdep, K, **kw)(x)
      x = jax.nn.sigmoid(x)
      x = x.reshape((*bshape, *x.shape[1:]))
      split = np.cumsum(
          [self.obs_space[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        out = embodied.jax.outs.MSE(out)
        out = embodied.jax.outs.Agg(out, 3, jnp.sum)
        recons[k] = out

    entries = {}
    return carry, entries, recons
