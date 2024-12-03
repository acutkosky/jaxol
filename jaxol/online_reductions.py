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
    return lambda grads, step_count, state, params: (beta, state)

def get_poly_beta_fn(p, c=1.0):
    return lambda grads, step_count, state, params: (jnp.clip(1.0-c*(step_count+1.0)**(-p), 0.0, 1.0), state)

def get_cosine_beta_fn(period=100):
    return lambda grads, step_count, state, params: (0.5-0.5*jnp.cos(step_count*2*jnp.pi/period), state)

def get_linear_decay_beta_fn(decay_end, decay_start=0):
    def beta_fn(grads, step_count, state, params):
        decay_beta = (decay_end - step_count) / (decay_end - decay_start)

        beta = 1.0 - jnp.minimum(1.0, decay_beta)

        return (beta, state)

    return beta_fn


def uniform_weight_fn(grads, step_count, state, params):
    return (1.0, state)


def linear_weight_ratio_fn(grads, step_count, state, params):
    return ((step_count + 1) / (step_count + 2), state)


def get_polynomial_weight_ratio_fn(power=1):
    def weight_ratio_fn(grads, step_count, state, params):
        return (
            (step_count + 1) ** power / (step_count + 2) ** power,
            state,
        )

    return weight_ratio_fn


def get_step_transform_weight_ratio_fn(tx: Callable[float, float] = lambda x: x**2):
    def weight_ratio_fn(grads, step_count, state, params):
        return (tx(step_count + 1) / tx(step_count + 2), state)

    return weight_ratio_fn


def get_ema_weight_ratio_fn(beta):
    def ema_weight_ratio_fn(grads, step_count, state, params):
        return (beta, state)

    return ema_weight_ratio_fn


def get_random_beta_fn(min_beta=0):
    def random_beta_fn(grads, step_count, state, params):
        to_use, rng = jr.split(state)
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
            grads, state.step_count, state.weight_state, params
        )

        next_averaging_factor = get_next_averaging_factor(
            next_weight_ratio, state.averaging_factor
        )

        # jax.debug.print("grads in averaging: {g}",g=grads)

        base_updates, next_base_state = base_optimizer.update(
            grads, state.base_state, params=params, next_weight_ratio=next_weight_ratio
        )

        beta, next_beta_state = beta_fn(grads, state.step_count, state.beta_state, params)

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
        # jax.debug.print("update norm: {u}", u=optax.tree_utils.tree_l2_norm(updates))

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


class O2NCState(NamedTuple):
    random_key: jax.Array
    base_state: optax.OptState
    step_count: jax.Array
    weight_state: Any


def o2nc(
    base_optimizer: OnlineLearner,
    random_key: jax.Array,
    next_weight_ratio_fn: Union[Callable, float] = 1.0,
    weight_state_init: Optional[Callable] = None,
):
    if not callable(next_weight_ratio_fn):
        next_weight_ratio_fn = get_ema_weight_ratio_fn(next_weight_ratio_fn)

    if not callable(weight_state_init):
        weight_state_init_value = weight_state_init
        weight_state_init = lambda params: weight_state_init_value

    def init_fn(params: optax.Params):
        return O2NCState(
            random_key, base_optimizer.init(params), 0, weight_state_init(params)
        )

    def update_fn(
        grads: optax.Updates, state: O2NCState, params: Optional[optax.Params] = None
    ):
        next_weight_ratio, next_weight_state = next_weight_ratio_fn(
            grads, state.step_count, state.weight_state, params
        )
        next_step_count = state.step_count + 1
        base_updates, next_base_state = base_optimizer.update(
            grads, state.base_state, next_weight_ratio, params, None
        )

        next_random_key, to_use = jax.random.split(state.random_key)

        scale = jax.random.exponential(to_use)

        updates = optax.tree_utils.tree_scalar_mul(scale, base_updates)

        next_state = O2NCState(
            next_random_key,
            next_base_state,
            next_step_count,
            next_weight_state,
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
    normalize=False,
    force_positive=True,
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
            prev_scale=jnp.zeros(1),
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

        # jax.debug.print("grads inn 1dred: {g}",g=grads)
        # direction_grad =
        direction_updates, next_direction_state = direction_learner.update(
            grads,
            state.direction_state,
            next_weight_ratio=next_weight_ratio,
            params=state.prev_direction,
            context=context,
        )
        direction = otu.tree_add(state.prev_direction, direction_updates)

        if normalize:
            direction_to_use = JOU.tree_l2_normalize(direction)
            prev_direction_to_use = JOU.tree_l2_normalize(state.prev_direction)
        else:
            direction_to_use = direction
            prev_direction_to_use = state.prev_direction

        # jax.debug.print("dir updates: {x}", x = direction_updates)

        scale_grads = otu.tree_vdot(prev_direction_to_use, grads)
        if force_positive:
            scale_grads = scale_grads * (state.prev_scale >= 0) + scale_grads * (
                state.prev_scale < 0
            ) * (scale_grads < 0)

        scale_updates, next_scale_state = scale_learner.update(
            scale_grads,
            state.scale_state,
            next_weight_ratio=next_weight_ratio,
            params=state.prev_scale,
            context=context,
        )
        scale = otu.tree_add(state.prev_scale, scale_updates)

        if force_positive:
            processed_scale = jnp.maximum(0.0, scale)
            prev_processed_scale = jnp.maximum(0.0, state.prev_scale)
        else:
            processed_scale = scale
            prev_processed_scale = state.prev_scale
            

        next_param = otu.tree_scalar_mul(processed_scale, direction_to_use)
        prev_param = otu.tree_scalar_mul(prev_processed_scale, prev_direction_to_use)

        updates = otu.tree_sub(next_param, prev_param)
        # jax.debug.print("direction: {x}", x=direction)
        # jax.debug.print("scale grad: {x}",x=scale_grads)
        # jax.debug.print("scale: {x}",x=scale)
        # jax.debug.print("nect_param: {x}, params: {y}",x=next_param,y=params)

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
    prev_weight_scale: float


def average_offset_ol(base_learner: OnlineLearner, grad_scale=False) -> OnlineLearner:
    base_learner = to_OL(base_learner)

    def init_fn(params):
        base_state = base_learner.init(params)
        offset = otu.tree_zeros_like(params)
        return AverageOffsetState(
            base_state=base_state,
            base_params=jtu.tree_map(jnp.array, params),  # copy params
            offset=offset,
            averaging_factor=1.0,
            prev_weight_scale=1.0,
        )

    def update_fn(
        grads: optax.Updates,
        state: AverageOffsetState,
        next_weight_ratio: jax.Array,
        params: optax.Params,
        context: Optional[Context] = None,
    ):
        base_updates, base_state = base_learner.update(
            grads,
            state.base_state,
            next_weight_ratio=next_weight_ratio,
            params=state.base_params,
            context=context,
        )
        # jax.debug.print("grads in offset: {g}",g=grads)
        if grad_scale:
            next_weight_scale = optax.tree_utils.tree_l2_norm(grads)**2 + 1e-8
        else:
            next_weight_scale = 1.0

        next_averaging_factor = get_next_averaging_factor(
            next_weight_ratio * state.prev_weight_scale / next_weight_scale,
            state.averaging_factor,
        )

        next_base_params = otu.tree_add(base_updates, state.base_params)

        def get_offset_update(o_i, p_i):
            return state.averaging_factor * (p_i - o_i)

        offset_update = jtu.tree_map(get_offset_update, state.offset, params)

        update = otu.tree_add(base_updates, offset_update)

        next_offset = otu.tree_add(state.offset, offset_update)

        next_state = AverageOffsetState(
            base_state=base_state,
            base_params=next_base_params,
            offset=next_offset,
            averaging_factor=next_averaging_factor,
            prev_weight_scale=next_weight_scale,
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
        state: optax.OptState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        new_state = []
        updates = grads
        for l, s in zip(learners, state):
            updates, learner_state = l.update(
                updates,
                s,
                next_weight_ratio=next_weight_ratio,
                params=params,
                context=context,
            )
            new_state.append(learner_state)

        return updates, new_state

    return OnlineLearner(init_fn, update_fn)


#### special learner that learns a direction, not actually a true online learner
class UnitDirectionLearnerState(NamedTuple):
    grad_sum: optax.Updates
    s_sum: jax.Array
    prev_s_sum: jax.Array


def unit_direction_learner(preprocess_grads=lambda g: g) -> OnlineLearner:
    def init_fn(params):
        return UnitDirectionLearnerState(
            grad_sum=otu.tree_zeros_like(params), s_sum=1.0, prev_s_sum=0.0,  # jnp.zeros(1)
        )

    def update_fn(
        grads: optax.Updates,
        state: UnitDirectionLearnerState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params]=None,
        context: Optional[Context] = None
    ):
        orig_grads = grads
        grads = preprocess_grads(grads)
        # jax.debug.print("grads: {g}",g=orig_grads)
        # jax.debug.print("grad norm: {n}, new norm: {p}",n=optax.tree_utils.tree_l2_norm(orig_grads), p=optax.tree_utils.tree_l2_norm(grads))
        next_grad_sum = jtu.tree_map(
            lambda s, g: (s + g)*next_weight_ratio, #get_next_accumulation(next_weight_ratio, s, g),
            state.grad_sum,
            grads,
        )
        next_s = (
            otu.tree_vdot(state.grad_sum, grads)
            / (otu.tree_l2_norm(state.grad_sum) + 1e-8)
            * jnp.sign(state.s_sum)
        )

        next_s_sum = jtu.tree_map(
            lambda old_sum, s: (old_sum + s)*next_weight_ratio, #get_next_accumulation(next_weight_ratio, old_sum, s),
            state.s_sum,
            next_s,
        )

        next_direction = otu.tree_scalar_mul(
            -jnp.sign(state.s_sum) / (otu.tree_l2_norm(next_grad_sum) + 1e-8),
            next_grad_sum,
        )
        # next_direction = jax.tree.map(lambda x: -x, next_direction)

        prev_direction = otu.tree_scalar_mul(
            -jnp.sign(state.prev_s_sum)/(otu.tree_l2_norm(state.grad_sum)+1e-8),
            state.grad_sum
        )


        

        updates = otu.tree_sub(next_direction, prev_direction)

        next_state = UnitDirectionLearnerState(
            grad_sum=next_grad_sum,
            s_sum=next_s_sum,
            prev_s_sum=state.s_sum,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


def add_learners(learners: List[OnlineLearner]):
    def init_fn(params: optax.Params):
        return [l.init(params) for l in learners]

    def update_fn(
        grads: optax.Updates,
        state: List[optax.OptState],
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params],
        context: Optional[Context] = None,
    ):
        next_state = []
        updates = optax.tree_utils.tree_zeros_like(grads)
        for i, s in enumerate(state):
            base_u, next_s = learners[i].update(
                grads, s, next_weight_ratio, params, context
            )
            next_state.append(next_s)
            updates = optax.tree_utils.tree_add_scalar_mul(
                updates, 1.0 / len(learners), base_u
            )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


def process_l2_constraint(unconstrained_params, grads, radius=1.0):
    param_norm = optax.tree_utils.tree_l2_norm(unconstrained_params)
    normalized_params = JOU.tree_l2_normalize(unconstrained_params)
    inner_product = optax.tree_utils.tree_vdot(normalized_params, grads)
    corrected_grads = jax.lax.select(
        jnp.logical_and(inner_product < 0, param_norm >= radius),
        optax.tree_utils.tree_add_scalar_mul(grads, -inner_product, normalized_params),
        grads,
    )

    return corrected_grads


class L2ConstraintState(NamedTuple):
    params: optax.Params
    base_state: optax.OptState


def l2_constraint(base_learner, radius=1.0):

    def init_fn(params: optax.Params):
        return L2ConstraintState(params=params, base_state=base_learner.init(params))

    def update_fn(
        grads: optax.Updates,
        state: L2ConstraintState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):

        grads = process_l2_constraint(state.params, grads, radius)
        base_updates, next_base_state = base_learner.update(
            grads, state.base_state, next_weight_ratio, params, context
        )
        next_params = optax.tree_utils.tree_add(state.params, base_updates)

        updates = optax.tree_utils.tree_sub(next_params, state.params)

        next_state = L2ConstraintState(params=next_params, base_state=next_base_state)
        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


class NoiseState(NamedTuple):
    rng_key: jax.Array


def add_noise(
    key: jax.Array,
    noise_scale: float = 1.0,
):

    def init_fn(params: optax.Params):
        return NoiseState(key)

    def update_fn(
        grads: optax.Updates, state: NoiseState, params: Optional[optax.Params] = None
    ):
        next_key, to_use = jax.random.split(state.rng_key)

        noise = optax.tree_utils.tree_random_like(to_use, grads, jax.random.normal)
        updates = optax.tree_utils.tree_add_scalar_mul(grads, noise_scale, noise)

        return updates, NoiseState(next_key)

    return optax.GradientTransformation(init_fn, update_fn)
