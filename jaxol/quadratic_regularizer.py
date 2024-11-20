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


def ftrl_learner(radius: Optional[jax.Array] = None) -> OnlineLearner:
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
            lambda s_i, g_i: get_next_accumulation(next_weight_ratio, s_i, g_i),
            state.grad_sum,
            grads,
        )

        next_grad_squared_sum = jtu.tree_map(
            lambda s_i, g_i: get_next_accumulation(
                next_weight_ratio**2, s_i, g_i**2
            ),
            state.grad_squared_sum,
            grads,
        )

        def get_update(next_sum, next_squared_sum, cur_sum, cur_squared_sum):
            cur_value = -cur_sum / jnp.sqrt(cur_squared_sum + 1e-8)
            if radius is not None:
                cur_value = jnp.clip(cur_value, -radius, radius)
            next_value = -next_sum / jnp.sqrt(next_squared_sum + 1e-8)
            if radius is not None:
                next_value = jnp.clip(next_value, -radius, radius)

            return next_value - cur_value

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
