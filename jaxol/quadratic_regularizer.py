import jax
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from optax import tree_utils as otu
import chex
import optax
from optax import Updates, Params, OptState, ScalarOrSchedule, GradientTransformation
from typing import Any, Tuple, NamedTuple, Optional, Union, Callable, Protocol
import sys
from jaxol.online_learner import (
    OnlineLearner,
    Context,
    to_OL,
    get_next_weight_ratio,
    get_next_averaging_factor,
    get_next_accumulation,
)


#### simple diagonal ftrl with quadratic regularizer
class FTRLState(NamedTuple):
    grad_sum: optax.Updates
    grad_squared_sum: optax.Updates


def ftrl_learner(lr: float = 1.0, radius: Optional[jax.Array] = None) -> OnlineLearner:
    def init_fn(params):
        return FTRLState(
            grad_sum=otu.tree_zeros_like(params),
            grad_squared_sum=otu.tree_zeros_like(params),
        )

    def update_fn(
        grads: optax.Updates,
        state: FTRLState,
        next_weight_ratio: jax.Array,
        params: optax.Params,
        context: Optional[Context] = None,
    ):
        next_grad_sum = jtu.tree_map(
            lambda s_i, g_i: (s_i + g_i) * next_weight_ratio,
            state.grad_sum,
            grads,
        )
        # next_grad_sum = jtu.tree_map(
        #     lambda s_i, g_i: get_next_accumulation(next_weight_ratio, s_i, g_i),
        #     state.grad_sum,
        #     grads,
        # )
        next_grad_squared_sum = jtu.tree_map(
            lambda s_i, g_i: (s_i + g_i**2) * next_weight_ratio**2,
            state.grad_squared_sum,
            grads,
        )
        # next_grad_squared_sum = jtu.tree_map(
        #     lambda s_i, g_i: get_next_accumulation(
        #         next_weight_ratio**2, s_i, g_i**2
        #     ),
        #     state.grad_squared_sum,
        #     grads,
        # )

        def get_update(next_sum, next_squared_sum, cur_sum, cur_squared_sum):
            cur_value = -cur_sum / jnp.sqrt(cur_squared_sum + 1e-8)
            if radius is not None:
                cur_value = jnp.clip(cur_value, -radius, radius)
            next_value = -next_sum / jnp.sqrt(next_squared_sum + 1e-8)
            if radius is not None:
                next_value = jnp.clip(next_value, -radius, radius)

            return lr * (next_value - cur_value)

        updates = jtu.tree_map(
            get_update,
            next_grad_sum,
            next_grad_squared_sum,
            state.grad_sum,
            state.grad_squared_sum,
        )

        next_state = FTRLState(
            grad_sum=next_grad_sum,
            grad_squared_sum=next_grad_squared_sum,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


#### laprop
class LaPropState(NamedTuple):
    grad_squared_sum: optax.Updates
    momentum: optax.Updates


def laprop_learner(beta1: float, beta2: float) -> OnlineLearner:
    def init_fn(params):
        return LaPropState(
            momentum=otu.tree_zeros_like(params),
            grad_squared_sum=otu.tree_zeros_like(params),
        )

    def update_fn(
        grads: optax.Updates,
        state: LaPropState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        next_grad_squared_sum = jtu.tree_map(
            lambda s_i, g_i: get_next_accumulation(beta2, s_i, (1 - beta2) * g_i**2),
            state.grad_squared_sum,
            grads,
        )

        def get_normalized_grad(g, next_squared_sum):
            return -g / jnp.sqrt(next_squared_sum + 1e-8)

        normalized_grad = jtu.tree_map(
            get_normalized_grad,
            grads,
            next_grad_squared_sum,
        )

        next_momentum = jtu.tree_map(
            lambda m_i, g_i: get_next_accumulation(beta1, m_i, g_i),
            state.momentum,
            normalized_grad,
        )

        next_state = LaPropState(
            momentum=next_momentum,
            grad_squared_sum=next_grad_squared_sum,
        )

        updates = next_momentum

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


##### mirror descent learners

class OGDState(NamedTuple):
    lr: jax.Array

def ogd_learner(lr: float) -> OnlineLearner:

    def init_fn(params):
        return OGDState(lr)

    def update_fn(
        grads: optax.Updates,
        state: OGDState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        updates = optax.tree_utils.tree_scalar_mul(-state.lr, grads)
        return updates, state

    return OnlineLearner(init_fn, update_fn)


class MDState(NamedTuple):
    grad_squared_sum: optax.Updates


def mirrordescent_learner(beta: float, lr_rescale=1.0) -> OnlineLearner:
    if lr_rescale == "beta":
        lr_rescale = 1.0 / jnp.sqrt(1.0 - beta)

    def init_fn(params):
        return MDState(grad_squared_sum=otu.tree_zeros_like(params))

    def update_fn(
        grads: optax.Updates,
        state: MDState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        if beta is not None:
            ratio = beta
        else:
            ratio = next_weight_ratio

        next_grad_squared_sum = jtu.tree_map(
            lambda s_i, g_i: get_next_accumulation(ratio, s_i, g_i**2),
            state.grad_squared_sum,
            grads,
        )

        def get_update(g, next_squared_sum):
            return -lr_rescale * g / jnp.sqrt(next_squared_sum + 1e-8)

        updates = jtu.tree_map(
            get_update,
            grads,
            next_grad_squared_sum,
        )

        next_state = MDState(
            grad_squared_sum=next_grad_squared_sum,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


class OptimisticMDState(NamedTuple):
    prev_grad: optax.Updates
    grad_squared_sum: optax.Updates
    averaging_factor: float


def optimisticmirrordescent_learner(beta: float) -> OnlineLearner:
    def init_fn(params):
        return OptimisticMDState(
            prev_grad=otu.tree_zeros_like(params),
            grad_squared_sum=otu.tree_zeros_like(params),
            averaging_factor=1.0,
        )

    def update_fn(
        grads: optax.Updates,
        state: OptimisticMDState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        if beta is not None:
            ratio = beta
        else:
            ratio = next_weight_ratio

        next_grad_squared_sum = jtu.tree_map(
            lambda s_i, g_i, p_i: get_next_accumulation(ratio, s_i, (g_i - p_i) ** 2),
            state.grad_squared_sum,
            grads,
            state.prev_grad,
        )

        def get_update(g, p, prev_squared_sum, next_squared_sum):
            current_update = -g / jnp.sqrt(g**2 + next_squared_sum + 1e-8)
            prev_update = -p / jnp.sqrt(p**2 + prev_squared_sum + 1e-8)
            return 2 * current_update - prev_update

        updates = jtu.tree_map(
            get_update,
            grads,
            state.prev_grad,
            state.grad_squared_sum,
            next_grad_squared_sum,
        )

        next_averaging_factor = get_next_averaging_factor(
            next_weight_ratio, state.averaging_factor
        )
        next_prev_grad = jtu.tree_map(
            lambda p_i, g_i: p_i + next_averaging_factor * (g_i - p_i),
            state.prev_grad,
            grads,
        )

        next_state = OptimisticMDState(
            prev_grad=next_prev_grad,
            averaging_factor=next_averaging_factor,
            grad_squared_sum=next_grad_squared_sum,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


class L2CorrelationLossState(NamedTuple):
    sum_squared_grad: jax.Array
    params: optax.Params
    sum_correlations: jax.Array


def l2_correlation_learner(lr=1.0, r_max=1.0, inner_product_scale=1.0, do_sqrt_scale=False):
    def init_fn(params: optax.Params):
        return L2CorrelationLossState(
            0.0,
            optax.tree_utils.tree_zeros_like(params),
            0.0,
        )

    def update_fn(
        grads: optax.Updates,
        state: L2CorrelationLossState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        # fixed_grads = optax.tree_utils.tree_scalar_mul(
        #     1 + optax.tree_utils.tree_vdot(grads, state.params) * inner_product_scale,
        #     grads,
        # )

        grad_norm_squared = optax.tree_utils.tree_l2_norm(grads, squared=True)
        # next_sum_squared_grad = (
        #     state.sum_squared_grad + optax.tree_utils.tree_vdot(grads, state.params)**2/r_max
        # ) * next_weight_ratio**2
        next_sum_squared_grad = (
            state.sum_squared_grad + grad_norm_squared
        ) * next_weight_ratio**2

        grad_param_cor = optax.tree_utils.tree_vdot(
            grads.flatten(), state.params.flatten()
        )
        next_sum_correlations = (
            state.sum_correlations + grad_param_cor**2
        ) * next_weight_ratio**2

        if do_sqrt_scale:
            reg_scaling = inner_product_scale / jnp.sqrt(next_sum_correlations + 1e-8)
        else:
            reg_scaling = inner_product_scale

        eta = lr / (jnp.sqrt(next_sum_squared_grad) + grad_norm_squared + 1e-8)
        fixed_eta = eta + reg_scaling * eta * (
            grad_param_cor + eta * grad_norm_squared
        ) / (1 + reg_scaling * grad_norm_squared)

        next_params = jax.tree.map(
            lambda g, p: p - fixed_eta * g,
            grads,
            state.params,
        )

        next_params_norm = optax.tree_utils.tree_l2_norm(next_params)
        next_params = optax.tree_utils.tree_scalar_mul(
            jnp.minimum(1.0, r_max / (next_params_norm + 1e-8)), next_params
        )
        # jax.debug.print("next norm: {x}", x=optax.tree_utils.tree_l2_norm(next_params))
        # jax.debug.print("prev norm: {x}", x=optax.tree_utils.tree_l2_norm(state.params))

        updates = optax.tree_utils.tree_sub(next_params, params)
        # jax.debug.print("udpates: {x}", x=updates)

        next_state = L2CorrelationLossState(
            next_sum_squared_grad, next_params, next_sum_correlations
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


class MaxLRState(NamedTuple):
    lr: jax.Array
    sum_grad: optax.Updates
    sum_loss: jax.Array
    sum_squared_grad: jax.Array
    iter_count: jax.Array
    last_iterate: optax.Params


def max_lr_learner(radius: float = 1.0, lr_init: float = 1.0):

    def init_fn(params: optax.Params):
        return MaxLRState(
            lr=lr_init,
            sum_grad=jax.tree.map(jnp.zeros_like, params),
            sum_loss=jnp.zeros([]),
            sum_squared_grad=jnp.zeros([]),
            iter_count=jnp.zeros([]),
            last_iterate=jax.tree.map(jnp.zeros_like, params),
        )

    def update_fn(
        grads: optax.Updates,
        state: MaxLRState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):

        next_sum_loss = (
            state.sum_loss + optax.tree_utils.tree_vdot(grads, state.last_iterate)
        ) * next_weight_ratio

        next_sum_squared_grad = (
            state.sum_squared_grad + optax.tree_utils.tree_l2_norm(grads, squared=True)
        ) * next_weight_ratio**2

        next_iter_count = state.iter_count + 1

        next_sum_grad = jax.tree.map(
            lambda s, g: (s + g) * next_weight_ratio, state.sum_grad, grads
        )

        regret = next_sum_loss + radius * optax.tree_utils.tree_l2_norm(next_sum_grad)

        ratio = jnp.maximum(
            1.0, regret / (radius * jnp.sqrt(2 * next_sum_squared_grad) + 1e-8)
        )

        next_lr = state.lr / ratio

        next_iterate = jax.tree.map(
            lambda x, g: x
            - g * next_lr / (jnp.sqrt(next_sum_squared_grad / next_iter_count) + 1e-8),
            state.last_iterate,
            grads,
        )

        next_iterate_norm = optax.tree_utils.tree_l2_norm(next_iterate)
        next_iterate = optax.tree_utils.tree_scalar_mul(
            jnp.minimum(1.0, radius / (next_iterate_norm + 1e-8)), next_iterate
        )

        next_state = MaxLRState(
            lr=next_lr,
            sum_grad=next_sum_grad,
            sum_loss=next_sum_loss,
            sum_squared_grad=next_sum_squared_grad,
            iter_count=next_iter_count,
            last_iterate=next_iterate,
        )

        updates = optax.tree_utils.tree_sub(next_iterate, state.last_iterate)

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)
