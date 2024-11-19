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

# Define the weight_ratio r_t = w_{t-1}/w_t ,  where w_t is the weight on the t^th iterate.
# So, if for any sequence x_t we define:
# S_T = sum_{t=1}^T w_t x_t / w_T
# then we have:
# S_T = S_{T-1}*r_T + x_T
# as an application, for  alpha_t = w_t/w_{1:t}, we have
# 1/alpha_t = 1/alpha_{t-1} * r_t + 1
# alpha_t = alpha_{t-1}/(r_t + alpha_{t-1})
# Also, if
# Z_T = sum_{t=1}^T w_t^p x_t/w_T^p
# then we have:
# Z_T = Z_{t-1} * r_T^p + x_t
# Therefore, for gamma_t = w_t^2/w^2_{1:t}, we have
# 1/gamma_t = 1/gamma_{t-1} * r_t^2 + 1
# gamma_t = gamma_{t-1}/(r_t^2 + gamma_{t-1})

# We will ask users to provide us with the next weight ratio r_{t+1} at time t.


# We define the averaging factor is defined as:
# alpha_t = w_t/w_{1:t}
# Notice that alpha_t is 1-beta_t for the familiar EMA beta.
# We can solve for the weight ratio as follows:
# r_t = alpha_{t-1}/alpha_t -1


def get_next_weight_ratio(averaging_factor, next_averaging_factor):
    return averaging_factor / next_averaging_factor - 1.0


def get_next_averaging_factor(next_weight_ratio, averaging_factor):
    return averaging_factor / (next_weight_ratio + averaging_factor)


def get_next_accumulation(next_weight_ratio, accumulation, next_value):
    return accumulation * next_weight_ratio + next_value


class OnlineLearner(optax.GradientTransformation):
    pass
    # init_fn: Callable
    # update_fn: Callable
    # def init(params: optax.Params):
    #     return self.init_fn(params)
    # def update(self, grad: optax.Updates, state: NamedTuple, *, weight_factor: jax.Array, params: optax.Params, **extra_kwargs):
    #     return self.update_fn(grad, state, weight_factor=weight_factor, params=params, **extra_kwargs)


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


def uniform_weight_fn(grads, state, params):
    return (1.0, state.weight_state)


def linear_weight_ratio_fn(grads, state, params):
    return (state.step_count / (state.step_count + 1), state.weight_state)


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

        # jax.debug.print(
        #     "next averaging factor:  {x}, weight ratio:  {y}",
        #     x=next_averaging_factor,
        #     y=next_weight_ratio,
        # )
        base_updates, next_base_state = base_optimizer.update(
            grads, state.base_state, params=params, next_weight_ratio=next_weight_ratio
        )

        beta, next_beta_state = beta_fn(grads, state, params)

        # def recover_iterate(p_i, avg_i):
        #     return (p_i - state.last_beta * avg_i) / (1 - state.last_beta)

        # base_iterate = jtu.tree_map(recover_iterate, params, state.average_iterate)

        # next_iterate = optax.apply_updates(base_iterate, base_updates)

        # def update_average_iterate(old_average_i, new_iterate_i):
        #     return old_average_i + (new_iterate_i - old_average_i) * next_averaging_factor

        # next_average_iterate = jtu.tree_map(
        #     update_average_iterate, state.average_iterate, base_iterate
        # )

        m_in_next_m_ratio = next_averaging_factor * (1.0 / state.averaging_factor - 1.0)

        def update_momentum(mi, base_ui):
            return m_in_next_m_ratio * mi + next_averaging_factor * base_ui

        next_momentum = jtu.tree_map(update_momentum, state.momentum, base_updates)

        # def get_updates(avg_i, base_i, p_i):
        #     return avg_i * beta + (1 - beta) * base_i - p_i

        # updates = jtu.tree_map(get_updates, next_average_iterate, next_iterate, params)

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


def GT_to_OL(tx: optax.GradientTransformation):
    def init_fn(params):
        return tx.init(params)

    def update_fn(
        grads: optax.Updates, state: NamedTuple, params: optax.Params, **extra_kwargs
    ):
        return tx.update(grads, state, params)

    return OnlineLearner(init_fn, update_fn)


# class GT_to_OL(OnlineLearner):
#     tx: optax.GradientTransformation

#     # def __init__(self, tx: optax.GradientTransformation):
#     #     self.tx = tx


#     def init(self, params: optax.Params):
#         return self.tx.init(params)
#     def update(self, grads: optax.Updates, state: NamedTuple, params: optax.Params, **extra_kwargs):
#         return self.tx.update(grads, state, params)
#         # updates, next_state = self.tx.update(grads, state, params)
#         # next_params = optax.apply_updates(params, updates)
#         # return next_params, next_state


class OLtoGTState(NamedTuple):
    ol_state: Any
    count: jax.Array


def OL_to_GT(ol: OnlineLearner):
    def init_fn(params):
        ol_state = ol.init(params)
        return OLtoGTState(ol_state=ol_state, count=0.0)

    def update_fn(grads, state, params, next_weight_ratio=None, **extra_kwargs):
        if next_weight_ratio is None:
            next_weight_ratio = 1.0
        ol_update, ol_state = ol.update(
            grads,
            state.ol_state,
            params=params,
            next_weight_ratio=next_weight_ratio,
            **extra_kwargs
        )
        return ol_update, ol_state
        # next_params, next_state = ol.update(grads, state, params=params, weight_factor=weight_factor, **extra_kwargs)
        # updates = otu.tree_sub(next_params, params)
        # return updates, next_state

    return optax.GradientTransformation(ol.init, update_fn)


class OneDReductionState(NamedTuple):
    # g_sum: optax.Updates
    # s_sum: optax.Updates
    scale_state: Any
    direction_state: Any
    prev_direction: optax.Updates
    prev_scale: jax.Array


def to_OL(tx: Any):
    if not isinstance(tx, OnlineLearner):
        if isinstance(tx, optax.GradientTransformation):
            return GT_to_OL(tx)
        else:
            raise ValueError("unknown tx type!")
    else:
        return tx


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
        *,
        params: optax.Params,
        next_weight_ratio: jax.Array,
        **extra_kwargs
    ):
        # next_g_sum = otu.tree_add_scalar_mul(state.g_sum, grads, weight_factor)
        # g_sum_norm = otu.tree_l2_norm(next_g_sum) + 1e-8

        # direction_grad =
        direction_updates, next_direction_state = direction_learner.update(
            grads,
            state.direction_state,
            params=state.prev_direction,
            next_weight_ratio=next_weight_ratio,
            **extra_kwargs
        )
        direction = otu.tree_add(state.prev_direction, direction_updates)

        scale_grads = otu.tree_vdot(state.prev_direction, grads)

        scale_updates, next_scale_state = scale_learner.update(
            scale_grads,
            state.scale_state,
            params=state.prev_scale,
            next_weight_ratio=next_weight_ratio,
            **extra_kwargs
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
        *,
        params: optax.Params,
        next_weight_ratio: jax.Array,
        **extra_kwargs
    ):
        base_updates, base_state = base_learner.update(
            grads,
            state.base_state,
            params=state.base_params,
            next_weight_ratio=next_weight_ratio,
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
        params: optax.Params,
        next_weight_ratio: jax.Array,
        **extra_kwargs
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
        state: OneDReductionState,
        *,
        params: optax.Params,
        next_weight_ratio: jax.Array,
        **extra_kwargs
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
        state: OneDReductionState,
        *,
        params: optax.Params,
        next_weight_ratio: jax.Array,
        **extra_kwargs
    ):
        next_grad_squared_sum = jtu.tree_map(
            lambda s_i, g_i: get_next_accumulation(beta2, s_i, (1-beta2)*g_i**2),
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
    if lr_rescale == 'beta':
        lr_rescale = 1.0/jnp.sqrt(1.0-beta)
    def init_fn(params):
        return MDState(grad_squared_sum=otu.tree_zeros_like(params))

    def update_fn(
        grads: optax.Updates,
        state: OneDReductionState,
        *,
        params: optax.Params,
        next_weight_ratio: jax.Array,
        **extra_kwargs
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
            return - lr_rescale  * g / jnp.sqrt(next_squared_sum + 1e-8)

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
        return OptimisticMDState(prev_grad=otu.tree_zeros_like(params), grad_squared_sum=otu.tree_zeros_like(params), averaging_factor=1.0)

    def update_fn(
        grads: optax.Updates,
        state: OneDReductionState,
        *,
        params: optax.Params,
        next_weight_ratio: jax.Array,
        **extra_kwargs
    ):
        if beta is not None:
            ratio = beta
        else:
            ratio = next_weight_ratio

        next_grad_squared_sum = jtu.tree_map(
            lambda s_i, g_i, p_i: get_next_accumulation(ratio, s_i, (g_i-p_i)**2),
            state.grad_squared_sum,
            grads,
            state.prev_grad
        )

        def get_update(g, p, prev_squared_sum, next_squared_sum):
            current_update = -g / jnp.sqrt(g**2 + next_squared_sum + 1e-8)
            prev_update = -p/jnp.sqrt(p**2 + prev_squared_sum + 1e-8)
            return 2*current_update - prev_update

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
            lambda p_i, g_i: p_i + next_averaging_factor * (g_i-p_i),
            state.prev_grad,
            grads
        )

        next_state = OptimisticMDState(
            prev_grad=next_prev_grad,
            averaging_factor=next_averaging_factor,
            grad_squared_sum=next_grad_squared_sum,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)
