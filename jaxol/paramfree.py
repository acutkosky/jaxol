import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from optax import tree_utils as otu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol
from tensorflow_probability.substrates import jax as tfp
import sys
import online_reductions as OR

class DownscaleEpsilonState(NamedTuple):
    base_state: optax.State
    reward: jax.Array
    max_ratio: jax.Array


def downscale_epsilon(
    base_learner: OR.OnlineLearner,
    epsilon: float = 1.0,
    scale_strategy: str = 'log'
):
    if scale_strategy == 'log':
        scale_fn = lambda r: 1.0/r
    elif scale_strategy == 'loglog':
        scale_fn = lambda r: 1.0/(r * jnp.log(1+1.0/r))
    elif scale_strategy == 'constant':
        scale_fn = lambda r: 1.0/(r* jnp.log(1.0 +1.0/r)**2)
    else:
        raise ValueError(f"unknown scale_strategy: {scale_strategy}")

    def init_fn(params: optax.Params):
        base_state = base_learner.init(params)
        reward = 0.0
        max_ratio = 0.0

        return DownscaleEpsilonState(
            base_state=base_state,
            reward=reward,
            max_ratio=max_ratio
        )

    
    


#### PDE-based algorithm from https://arxiv.org/abs/2309.16044 ###
# We make one small change: rather than Phi(V,S) = phi(V + z + kS, S),
# we do Phi(V, S) = phi(V + z + k|S|, S).
# This seems to allow us to not need to restrict to positive predictions
# for the 1-d algorithm.

def scaled_erfi_integral(z):
    # computes integral scaled_erfi(z) dz
    # scaled_erfi(x) = sqrt(pi)/2 * erfi(x)
    # from https://mathworld.wolfram.com/Erfi.html
    # we have that
    # int erfi(z) dz = z erfi(z) - exp(z**2)/sqrt(pi)
    # so
    # int scaled_erfi(z) dz = sqrt(pi)/2 z * erfi(z) - exp(z**2)/2
    #                       = z * scaled_erfi(z) - exp(z**2)/2
    return z * scaled_erfi(z) - jnp.exp(z**2) / 2


def scaled_erfi(z):
    # computes sqrt(pi)/2 *  erfi(z) = int_0^z exp(t**2) dt
    #
    # see https://github.com/jax-ml/jax/issues/9098
    # and https://mathworld.wolfram.com/DawsonsIntegral.html
    # and https://www.tensorflow.org/probability/api_docs/python/tfp/math/dawsn
    # from which we have:
    # dawson's integral(z) = exp(-z**2) * scaled_erfi(z)
    return tfp.math.dawsn(z) * jnp.exp(z**2)


def littlephi(x, y, eps, alpha):
    return (
        eps
        * jnp.sqrt(alpha * x)
        * (2 * scaled_erfi_integral(y / jnp.sqrt(4 * alpha * x)) - 1)
    )


def bigphi(V, S, z, k):
    return littlephi(V + z + k * jnp.abs(S), S)


def d2_bigphi(V, S, z, k):
    # I will assume that jax.grad is just as good as manually implementing
    # the derivative until someone tells me otherwise.
    return jax.grad(lambda s: bigphi(V, s, z, k))(S)


class RefinedPDEPerCoordState(NamedTuple):
    V: jax.Array
    S: jax.Array
    h: jax.Array


def refined_pde_per_coord(
    eps: jax.Array = 1.0, G: jax.Array = 1e-8, alpha: jax.Array = 1.0
):
    def init_fn(params: optax.Params):
        V = jax.tree.map(jnp.zeros_like, params)
        S = jax.tree.map(jnp.zeros_like, params)
        h = jax.tree.map(lambda p: jnp.full_like(p, fill_value=G), params)
        return RefinedPDEPerCoordState(
            V=V,
            S=S,
            h=h,
        )

    def update_fn(
        grads: optax.Updates,
        state: RefinedPDEPerCoordState,
        *,
        params: optax.Params,
        next_weight_ratio: jax.Array,
        **extra_kwargs
    ):
        # h_t is max_{i< t} w_i |g_i|/w_{t}
        # weight ratio r is w_t/w_{t+1}

        def get_z(h):
            return jax.tree.map(
                lambda h_: (12 * alpha + 4) / (2 * alpha - 1) * h_**2, h
            )

        def get_k(h):
            return jax.tree.map(lambda h_: 2 * h_, h)

        h_t = state.h
        z_t = get_z(h_t)
        k_t = get_k(h_t)

        h_t_plus_one = jax.tree.map(
            lambda g_, h_: jnp.maximum(jnp.abs(g_), h_) * next_weight_ratio, grads, h_t
        )
        z_t_plus_one = get_z(h_t_plus_one)
        k_t_plus_one = get_k(h_t_plus_one)

        # V_t is sum_{i=1}^t g_i^2 * w_i^2 / w_{t+1}^2

        V_t_minus_one = state.V

        V_t = jax.tree.map(
            lambda g_, v_: (v_ + g_**2) * next_weight_ratio**2, grads, V_t_minus_one
        )

        # S_t is sum_{i=1}^t g_i * w_i / w_{t+1}

        S_t_minus_one = state.S

        S_t = jax.tree.map(
            lambda g_, s_: (s_ - g_) * next_weight_ratio, grads, S_t_minus_one
        )

        unscaled_prev_params = jax.tree_map(
            d2_bigphi,
            V_t_minus_one,
            S_t_minus_one,
            z_t,
            k_t,
        )

        unscaled_next_params = jax.tree.map(
            d2_bigphi,
            V_t,
            S_t,
            z_t_plus_one,
            k_t_plus_one,
        )

        updates = jax.tree_map(
            lambda n_, p_: eps * (n_ - p_), unscaled_next_params, unscaled_prev_params
        )

        next_state = RefinedPDEPerCoordState(V=V_t, S=S_t, h=h_t_plus_one)

        return updates, next_state

    return OR.OnlineLearner(init_fn, update_fn)
