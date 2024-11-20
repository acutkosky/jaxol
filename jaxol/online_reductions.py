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


class GeneralizedAveragingState(NamedTuple):
    momentum: optax.Params
    # average_iterate: optax.Params
    base_state: optax.OptState
    step_count: jax.Array
    beta_state: Any
    weight_state: Any
    averaging_factor: jax.Array
    last_beta: jax.Array


def get_constant_beta_fn(beta):
    return lambda grads, state, params: (beta, state.beta_state)


def get_linear_decay_beta_fn(decay_end, decay_start=0):
    def beta_fn(grads, state, params):
        decay_beta = (decay_end - state.step_count) / (decay_end - decay_start)

        beta = 1.0 - jnp.minimum(1.0, decay_beta)

        return (beta, state.weight_state)

    return beta_fn


def uniform_weight_fn(grads, state, params):
    return (1.0, state.weight_state)


def linear_weight_ratio_fn(grads, state, params):
    return ((state.step_count + 1) / (state.step_count + 2), state.weight_state)


def get_polynomial_weight_ratio_fn(power=1):
    def weight_ratio_fn(grads, state, params):
        return (
            (state.step_count + 1) ** power / (state.step_count + 2) ** power,
            state.weight_state,
        )

    return weight_ratio_fn


def get_step_transform_weight_ratio_fn(tx: Callable[float, float] = lambda x: x**2):
    def weight_ratio_fn(grads, state, params):
        return (tx(state.step_count + 1) / tx(state.step_count + 2), state.weight_state)

    return weight_ratio_fn


def get_ema_weight_ratio_fn(beta):
    def ema_weight_ratio_fn(grads, state, params):
        return (beta, state.weight_state)

    return ema_weight_ratio_fn


def get_random_beta_fn(min_beta=0):
    def random_beta_fn(grads, state, params):
        to_use, rng = jr.split(state.beta_state)

        beta = jr.uniform(to_use, minval=min_beta)
        return beta, rng

    return random_beta_fn


def get_inference_parameters(optimizer_state):
    averaging_states = jtu.tree_leaves(
        optimizer_state, is_leaf=lambda n: isinstance(n, GeneralizedAveragingState)
    )
    averaging_states = [
        x for x in averaging_states if isinstance(x, GeneralizedAveragingState)
    ]
    return averaging_states[0].average_iterate


def generalized_averaging(
    base_optimizer: OnlineLearner,
    beta_fn: Union[Callable, float] = 1.0,
    beta_state_init: Optional[Callable] = None,
    next_weight_ratio_fn: Union[Callable, float] = 1.0,
    weight_state_init: Optional[Callable] = None,
) -> optax.GradientTransformation:
    """
    arguments:
    base_optimizer: online learner to convert into a stochastic optimization algorithm.
    beta_fn: function that determines how to interpolate between different averaging schemes.
        beta_fn(grads: optax.Updates, state: GeneralizedAveragingState, params: optax.Params) -> (beta: float, next_beta_state: Any)

    beta_state_init: provides the initial state for beta_fn. (optax.Params) -> Any
    next_weight_ratio_fn: compute the weight ratio w_t/w_{t+1}.
        next_ratio_ratio_fn(grads: optax.Updates, state: GeneralizedAveragingState, params: optax.Params) -> (next_ratio_ratio: float, next_weight_state: Any)
    weight_state_init: provides the  initial state for the next_weight_ratio_fn. (optax.Params) -> Any
    """
    base_optimizer = to_OL(base_optimizer)

    if not callable(beta_fn):
        beta = beta_fn
        beta_fn = get_constant_beta_fn(beta)

    if not callable(beta_state_init):
        beta_state_init_value = beta_state_init
        beta_state_init = lambda params: beta_state_init_value

    if not callable(next_weight_ratio_fn):
        next_weight_ratio_fn = get_ema_weight_ratio_fn(next_weight_ratio_fn)

    if not callable(weight_state_init):
        weight_state_init_value = weight_state_init
        weight_state_init = lambda params: weight_state_init_value

    def init_fn(params: optax.Params):
        base_state = base_optimizer.init(params)
        # average_iterate = jtu.tree_map(jnp.zeros_like, params)
        momentum = jtu.tree_map(jnp.zeros_like, params)
        weight_state = weight_state_init(params)
        beta_state = beta_state_init(params)

        return GeneralizedAveragingState(
            momentum=momentum,
            # average_iterate=average_iterate,
            base_state=base_state,
            step_count=0,
            beta_state=beta_state,
            weight_state=weight_state,
            averaging_factor=1.0,
            last_beta=0.0,
        )

    def update_fn(
        grads: optax.Updates, state: GeneralizedAveragingState, params: optax.Params
    ):
        next_weight_ratio, next_weight_state = next_weight_ratio_fn(
            grads, state, params
        )

        next_averaging_factor = get_next_averaging_factor(
            next_weight_ratio, state.averaging_factor
        )

        base_updates, next_base_state = base_optimizer.update(
            grads, state.base_state, params=params, next_weight_ratio=next_weight_ratio
        )

        beta, next_beta_state = beta_fn(grads, state, params)

        m_in_next_m_ratio = next_averaging_factor * (1.0 / state.averaging_factor - 1.0)

        def update_momentum(mi, base_ui):
            return m_in_next_m_ratio * mi + next_averaging_factor * base_ui

        next_momentum = jtu.tree_map(update_momentum, state.momentum, base_updates)

        m_in_update_ratio = state.last_beta + (state.last_beta - beta) * (
            1.0 / next_averaging_factor - 1.0
        )

        def get_updates(m_i, base_i):
            return m_in_update_ratio * m_i + (1.0 - state.last_beta) * base_i

        updates = jtu.tree_map(get_updates, next_momentum, base_updates)

        next_state = GeneralizedAveragingState(
            momentum=next_momentum,
            # average_iterate=next_average_iterate,
            base_state=next_base_state,
            step_count=state.step_count + 1,
            beta_state=next_beta_state,
            weight_state=next_weight_state,
            averaging_factor=next_averaging_factor,
            last_beta=beta,
        )

        return updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


class OneDReductionState(NamedTuple):
    # g_sum: optax.Updates
    # s_sum: optax.Updates
    scale_state: Any
    direction_state: Any
    prev_direction: optax.Updates
    prev_scale: jax.Array


def one_d_reduction(
    scale_learner: OnlineLearner,
    direction_learner: OnlineLearner,
) -> OnlineLearner:
    direction_learner = to_OL(direction_learner)
    scale_learner = to_OL(scale_learner)

    def init_fn(params: optax.Params):
        # g_sum = otu.tree_zeros_like(params)
        # s_sum = jnp.zeros(1)
        scale_state = scale_learner.init(jnp.zeros(1))
        direction_state = direction_learner.init(params)

        return OneDReductionState(
            # g_sum=g_sum,
            # s_sum=s_sum,
            scale_state=scale_state,
            direction_state=direction_state,
            prev_direction=otu.tree_zeros_like(params),
            prev_scale=jax.zeros(1),
        )

    def update_fn(
        grads: optax.Updates,
        state: OneDReductionState,
        next_weight_ratio: jax.Array,
        params: optax.Params,
        context: Optional[Context] = None,
    ):
        # next_g_sum = otu.tree_add_scalar_mul(state.g_sum, grads, weight_factor)
        # g_sum_norm = otu.tree_l2_norm(next_g_sum) + 1e-8

        # direction_grad =
        direction_updates, next_direction_state = direction_learner.update(
            grads,
            state.direction_state,
            next_weight_ratio=next_weight_ratio,
            params=state.prev_direction,
            context=context,
        )
        direction = otu.tree_add(state.prev_direction, direction_updates)

        scale_grads = otu.tree_vdot(state.prev_direction, grads)

        scale_updates, next_scale_state = scale_learner.update(
            scale_grads,
            state.scale_state,
            next_weight_ratio=next_weight_ratio,
            params=state.prev_scale,
            context=context,
        )
        scale = otu.tree_add(state.prev_scale, scale_updates)

        next_param = otu.tree_scalar_mul(scale, direction)

        updates = otu.tree_sub(next_param, params)

        next_state = OneDReductionState(
            scale_state=next_scale_state,
            direction_state=next_direction_state,
            prev_direction=direction,
            prev_scale=scale,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


class AverageOffsetState(NamedTuple):
    base_state: Any
    base_params: optax.Params
    offset: optax.Params
    averaging_factor: float


def average_offset_ol(
    base_learner: OnlineLearner,
) -> OnlineLearner:
    base_learner = to_OL(base_learner)

    def init_fn(params):
        base_state = base_learner.init(params)
        offset = otu.tree_zeros_like(params)
        return AverageOffsetState(
            base_state=base_state,
            base_params=jtu.tree_map(jnp.array, params),  # copy params
            offset=offset,
            averaging_factor=1.0,
        )

    def update_fn(
        grads: optax.Updates,
        state: AverageOffsetState,
        next_weight_ratio: jax.Array,
        params: optax.Params,
        context: Optional[Context],
    ):
        base_updates, base_state = base_learner.update(
            grads,
            state.base_state,
            next_weight_ratio=next_weight_ratio,
            params=state.base_params,
            context=context,
        )

        next_averaging_factor = get_next_averaging_factor(
            next_weight_ratio, state.averaging_factor
        )

        next_base_params = otu.tree_add(base_updates, state.base_params)

        def get_offset_update(o_i, p_i):
            return next_averaging_factor * (p_i - o_i)

        offset_update = jtu.tree_map(get_offset_update, state.offset, params)

        update = otu.tree_add(base_updates, offset_update)

        next_offset = otu.tree_add(state.offset, offset_update)

        next_state = AverageOffsetState(
            base_state=base_state,
            base_params=next_base_params,
            offset=next_offset,
            averaging_factor=next_averaging_factor,
        )

        return update, next_state

    return OnlineLearner(init_fn, update_fn)


def chain(*learners):
    learners = [to_OL(l) for l in learners]

    def init_fn(params):
        state = [l.init(params) for l in learners]
        return state

    def update_fn(
        grads: optax.Updates,
        state: OneDReductionState,
        *,
        params: optax.Params,
        next_weight_ratio: jax.Array,
        **extra_kwargs
    ):
        new_state = []
        updates = grads
        for l, s in zip(learners, state):
            updates, learner_state = l.update(
                updates,
                s,
                params=params,
                next_weight_ratio=next_weight_ratio,
                **extra_kwargs
            )
            new_state.append(learner_state)

        return updates, new_state

    return OnlineLearner(init_fn, update_fn)


#### special learner that learns a direction, not actually a true online learner
class UnitDirectionLearnerState(NamedTuple):
    grad_sum: optax.Updates
    s_sum: jax.Array


def unit_direction_learner() -> OnlineLearner:
    def init_fn(params):
        return UnitDirectionLearnerState(
            grad_sum=otu.tree_zeros_like(params), s_sum=jnp.zeros(1)
        )

    def update_fn(
        grads: optax.Updates,
        state: UnitDirectionLearnerState,
        *,
        next_weight_ratio: jax.Array,
        params: optax.Params,
        context: Optional[Context] = None
    ):
        next_grad_sum = jtu.tree_map(
            lambda s, g: get_next_accumulation(next_weight_ratio, s, g),
            state.grad_sum,
            grads,
        )
        next_s = (
            otu.tree_vdot(state.grad_sum, grads)
            / (otu.tree_l2_norm(state.grad_sum) + 1e-8)
            * jnp.sign(state.s_sum)
        )

        next_s_sum = jtu.tree_map(
            lambda old_sum, s: get_next_accumulation(
                next_weight_ratio, old_sum, s
            ).state.s_sum,
            next_s,
        )

        next_direction = otu.tree_scalar_mul(
            jnp.sign(state.s_sum) / (otu.tree_l2_norm(next_grad_sum) + 1e-8),
            next_grad_sum,
        )

        updates = otu.tree_sub(next_direction - params)

        next_state = UnitDirectionLearnerState(
            grad_sum=next_grad_sum,
            s_sum=next_s_sum,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)
