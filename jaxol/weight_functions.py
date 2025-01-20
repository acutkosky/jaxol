import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from optax import tree_utils as otu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol, List
import sys
from jaxol.online_learner import (
    OnlineLearner,
    Context,
    to_OL,
    get_next_weight_ratio,
    get_next_averaging_factor,
    get_next_accumulation,
)
import jaxol.utils as JOU



def get_poly_beta_fn(p, c=1.0):
    return lambda grads, state, next_weight_ratio, params: (
        jnp.clip(1.0 - c * (state.step_count + 1.0) ** (-p), 0.0, 1.0),
        state.beta_state,
    )


def get_cosine_beta_fn(period=100):
    return lambda grads, state, next_weight_ratio, params: (
        0.5 - 0.5 * jnp.cos(state.step_count * 2 * jnp.pi / period),
        state.beta_state,
    )


class SumWeightedPowGradBetaState(NamedTuple):
    sum_pow_grad: float

def get_sum_weighted_pow_grad_beta_fn(p=2.0):
    def beta_fn(grads, state, next_weight_ratio, params):
        sum_pow_grad = state.beta_state.sum_pow_grad
        norm_grad = optax.tree_utils.tree_l2_norm(grads)
        next_sum_pow_grad = (sum_pow_grad + norm_grad**p) * next_weight_ratio*p
        next_state = SumWeightedPowGradBetaState(next_sum_pow_grad)

        beta = jnp.maximum(0.0, 1.0-next_weight_ratio * norm_grad*(next_sum_pow_grad + 1e-8)**(1.0/p))

        return beta, next_state
    return beta_fn

class ParamDistBetaState(NamedTuple):
    init_params: jax.Array
    sum_squared_weight: float
    sum_weight: float

def param_dist_beta_init(params):
    return ParamDistBetaState(
        params,
        0.0,
        0.0
    )

def param_dist_beta_fn(grads, state, next_weight_ratio, params):
    # S_T = w^2_{1:T}/w_{T+1}^2
    # S_T = (S_{T-1} * w^2_T/w^2_{T+1} + 1)

    # Z_T = w_{1:T}/w_{T+1}
    # Z_T = (Z_{T-1} * w_T/w_{T+1} +  1)

    # beta = 1 - sqrt(S_T)/Z_T

    next_sum_squared_weight = state.beta_state.sum_squared_weight * next_weight_ratio + 1
    next_sum_weight = state.beta_state.sum_weight * next_weight_ratio + 1


    dist = optax.tree_utils.tree_l2_norm(
        optax.tree_utils.tree_sub(params, state.beta_state.init_params)
    )
    
    momentum = state.momentum
    averaging_factor = state.averaging_factor

    displacement = jax.tree.map(
        lambda m: m/averaging_factor,
        momentum
    )

    displacement_norm = optax.tree_utils.tree_l2_norm(
        displacement
    )
    dist_ratio = dist/(displacement_norm + 1e-8)
    
    beta = jnp.maximum(0.0, 1.0-dist_ratio*jnp.sqrt(next_sum_squared_weight)/next_sum_weight)

    next_state = ParamDistBetaState(
        state.beta_state.init_params,
        next_sum_squared_weight,
        next_sum_weight)
    return beta, next_state

   

class InvWeightBetaState(NamedTuple):
    sum_weight: float



def inv_weight_beta_fn(grads, state, next_weight_ratio, params):

    
    # S_T = w^2_{1:T}/w_{T+1}^2
    # S_T = (S_{T-1} * w^2_T/w^2_{T+1} + 1)

    # Z_T = w_{1:T}/w_{T+1}
    # Z_T = (Z_{T-1} * w_T/w_{T+1} +  1)

    # beta = 1 - sqrt(S_T)/Z_T

    next_sum_weight = state.beta_state.sum_weight * next_weight_ratio + 1

    beta = jnp.maximum(0.0, 1.0-1.0/next_sum_weight)

    next_state = InvWeightBetaState(
        next_sum_weight)
    return beta, next_state
class AvgSqrtWeightBetaState(NamedTuple):
    sum_squared_weight: float
    sum_weight: float

def avg_sqrt_weight_beta_fn(grads, state, next_weight_ratio, params):

    
    # S_T = w^2_{1:T}/w_{T+1}^2
    # S_T = (S_{T-1} * w^2_T/w^2_{T+1} + 1)

    # Z_T = w_{1:T}/w_{T+1}
    # Z_T = (Z_{T-1} * w_T/w_{T+1} +  1)

    # beta = 1 - sqrt(S_T)/Z_T

    next_sum_squared_weight = state.beta_state.sum_squared_weight * next_weight_ratio + 1
    next_sum_weight = state.beta_state.sum_weight * next_weight_ratio + 1

    beta = jnp.maximum(0.0, 1.0-jnp.sqrt(next_sum_squared_weight)/next_sum_weight)

    next_state = AvgSqrtWeightBetaState(
        next_sum_squared_weight,
        next_sum_weight)
    return beta, next_state
    
class SumPowGradBetaState(NamedTuple):
    sum_pow_grad: float

def get_sum_pow_grad_beta_fn(p=2.0):
    def beta_fn(grads, state, next_weight_ratio, params):
        sum_pow_grad = state.beta_state.sum_pow_grad
        norm_grad = optax.tree_utils.tree_l2_norm(grads)
        next_sum_pow_grad = sum_pow_grad + norm_grad**p
        next_state = SumPowGradBetaState(next_sum_pow_grad)

        beta = jnp.maximum(0.0, 1.0-(norm_grad)*(next_sum_pow_grad + 1e-8)**(1.0/p))

        return beta, next_state
    return beta_fn

def get_linear_decay_beta_fn(decay_end, decay_start=0):
    def beta_fn(grads, state, next_weight_ratio, params):
        decay_beta = (decay_end - state.step_count) / (decay_end - decay_start)

        beta = 1.0 - jnp.minimum(1.0, decay_beta)

        return (beta, state.beta_state)

    return beta_fn

class AcceleratedBetaState(NamedTuple):
    alpha: float

def get_accelerated_beta_fn(coef=1.0, p=1.0):
    def beta_fn(grads, state, next_weight_ratio, params):
        alpha = state.beta_state.alpha
        # alpha =  w^p_{1:t}/w_t^p
        next_alpha = alpha * next_weight_ratio**p + 1
        beta = jnp.maximum(0.0, 1.0 - coef * next_alpha**(1.0/p))
        next_state = AcceleratedBetaState(alpha)

        return beta, next_state
    return beta_fn

def get_capped_tail_polynomial_weight_ratio_fn(power=0.5, cap=1.0):
    def weight_ratio_fn(grads, state, params):
        return (
            jnp.minimum(1.0 - 1.0/(state.step_count+2) ** power, cap),
            state.weight_state
        )

    return weight_ratio_fn

def get_capped_polynomial_weight_ratio_fn(power=0.0, cap=1.0):
    def weight_ratio_fn(grads, state, params):
        return (
            jnp.minimum(1- power/(state.step_count + 1 + power), cap),
            state.weight_state,
        )

    return weight_ratio_fn

class ParamDistanceWeightRatioState(NamedTuple):
    init_params: jax.Array
    prev_dist: float

def param_dist_weight_ratio_init_fn(params):
    return ParamDistanceWeightRatioState(params, 0.0)

def get_exp_param_dist_weight_ratio_fn(p=0.0, force_positive=False):
    def weight_ratio_fn(grads, state, params):
        dist = optax.tree_utils.tree_l2_norm(optax.tree_utils.tree_sub(params, state.weight_state.init_params))
        if force_positive:
            dist = jnp.maximum(dist, state.weight_state.prev_dist)
        weight_ratio = jnp.exp(state.weight_state.prev_dist - dist) * ((1.0+state.step_count)/(state.step_count+2.0))**p
        next_state = ParamDistanceWeightRatioState(
            init_params=state.weight_state.init_params,
            prev_dist=dist
        ) 
        return weight_ratio, next_state
    return weight_ratio_fn

class OLAvgDistWeightRatioState(NamedTuple):
    prev_dist: float

def get_ol_avg_dist_weight_ratio_fn(p=0.0, force_positive=False, ignore_negative=False):

    def weight_ratio_fn(grads, state, params):
        prev_dist = state.weight_state.prev_dist
        momentum = state.momentum
        averaging_factor = state.averaging_factor

        displacement = jax.tree.map(
            lambda m: m/averaging_factor,
            momentum
        )

        dist = optax.tree_utils.tree_l2_norm(displacement)
        if force_positive:
            dist = jnp.maximum(prev_dist, dist)
        

        ratio = jnp.exp(prev_dist-dist) * ((1.0+state.step_count)/(2.0+state.step_count))**p
        if ignore_negative:
            ratio = jnp.minimum(ratio, 1.0)
        next_state = OLAvgDistWeightRatioState(
            dist
        )
        return ratio, next_state

    return weight_ratio_fn
class ExpGradSumWeightRatioState(NamedTuple):
    prev_weight_ratio: float
    sum_grad: float
    sum_squared_grad: float

def exp_grad_sum_weight_ratio_init_fn(params):
    return ExpGradSumWeightRatioState(
        1.0,
        optax.tree_utils.tree_zeros_like(params),
        0.0
    )

def exp_grad_sum_weight_ratio_fn(grads, state, params):

    prev_weight_ratio = state.weight_state.prev_weight_ratio

    next_sum_grad = jax.tree.map(
        lambda s, g: s*prev_weight_ratio + g,
        state.weight_state.sum_grad,
        grads
    )

    next_sum_squared_grad = state.weight_state.sum_squared_grad * prev_weight_ratio**2 + optax.tree_utils.tree_l2_norm(grads, squared=True)

    norm_next_sum = optax.tree_utils.tree_l2_norm(next_sum_grad)
    norm_sum = optax.tree_utils.tree_l2_norm(state.weight_state.sum_grad)
    
    log_weight_ratio = norm_next_sum/jnp.sqrt(next_sum_squared_grad+1e-8) - norm_sum/jnp.sqrt(state.weight_state.sum_squared_grad+1e-8)
    next_weight_ratio = jnp.exp(-log_weight_ratio)

    next_state = ExpGradSumWeightRatioState(
        next_weight_ratio,
        next_sum_grad,
        next_sum_squared_grad
    )

    return next_weight_ratio, next_state

class RestartingWeightRatioState(NamedTuple):
    ratio: float

def get_restart_weight_ratio_fn(coefficient=0.5):
    def weight_ratio_fn(grads, state, params):
        correlation = (optax.tree_utils.tree_vdot(JOU.tree_l2_normalize(grads), JOU.tree_l2_normalize(state.updates)))**2
        ratio = 1.0 - jnp.maximum(0.0, correlation)*coefficient
        next_state = RestartingWeightRatioState(
            ratio
        )

        # jax.debug.print("ratio: {r}", r=ratio)

        return ratio, next_state

    return weight_ratio_fn

class CorrelationWeightRatioState(NamedTuple):
    avg_correlation: float
    sum_squared_grad: float
    prev_squared_norm_grad: float
    ratio: float

def update_correlation_weight_ratio_fn(grads, state, params):
    correlation = (optax.tree_utils.tree_vdot(grads, JOU.tree_l2_normalize(state.updates)))**2

    avg_correlation, sum_squared_grad, prev_squared_norm_grad, prev_ratio = state.weight_state

    squared_norm_grad = optax.tree_utils.tree_l2_norm(grads, squared=True)
    next_sum_squared_grad = sum_squared_grad + squared_norm_grad
    
    next_avg_correlation = avg_correlation + (
        correlation  - avg_correlation
    ) / (state.step_count + 1)

    prev_weight = 1.0/(avg_correlation+1e-8) + prev_squared_norm_grad/(sum_squared_grad+1e-8)
    next_weight = 1.0/(next_avg_correlation+1e-8) + squared_norm_grad/(next_sum_squared_grad +1e-8)

    ratio = jnp.minimum(1.0, prev_weight/next_weight)

    # jax.debug.print("ratio: {r}",r=ratio)
    next_state = CorrelationWeightRatioState(
        avg_correlation=next_avg_correlation,
        sum_squared_grad=next_sum_squared_grad,
        prev_squared_norm_grad=squared_norm_grad,
        ratio=ratio
    )
    return ratio, next_state


def uniform_weight_fn(grads, state, params):
    return (1.0, state.weight_state)


def linear_weight_ratio_fn(grads, state, params):
    result = ((state.step_count + 1) / (state.step_count + 2), state.weight_state)
    return result


def get_polynomial_weight_ratio_fn(power=1):
    def weight_ratio_fn(grads, state, params):
        return (
            1- power/(state.step_count + 1 + power),
            state.weight_state,
        )

    return weight_ratio_fn


def get_step_transform_weight_ratio_fn(tx: Callable[float, float] = lambda x: x**2):
    def weight_ratio_fn(grads, state, params):
        return (tx(state.step_count + 1) / tx(state.step_count + 2), state.weight_state)

    return weight_ratio_fn


def get_random_beta_fn(min_beta=0):
    def random_beta_fn(grads, step_count, state, beta_state, params):
        to_use, rng = jr.split(state)
        beta = jr.uniform(to_use, minval=min_beta)
        return beta, rng

    return random_beta_fn

