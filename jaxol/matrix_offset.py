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
from jaxol.approx_prec import (
    get_prec_axes,
    multiply,
    make_matrix,
    restore_shape
)

class MatrixAverageOffsetState(NamedTuple):
    base_state: Any
    base_params: optax.Params
    offset: optax.Params


def matrix_average_offset_ol(base_learner: OnlineLearner) -> OnlineLearner:
    base_learner = to_OL(base_learner)

    def init_fn(params):
        base_state = base_learner.init(params)
        offset = otu.tree_zeros_like(params)
        return MatrixAverageOffsetState(
            base_state=base_state,
            base_params=jtu.tree_map(jnp.array, params),  # copy params
            offset=offset,
        )

    def update_fn(
        grads: optax.Updates,
        state: MatrixAverageOffsetState,
        next_weight_ratio: jax.Array,
        params: optax.Params,
        context: Optional[Context]=None,
    ):

        if context is None:
            context = {}

        base_updates, base_state = base_learner.update(
            grads,
            state.base_state,
            next_weight_ratio=next_weight_ratio,
            params=state.base_params,
            context=context,
        )
        weight = context['prec_increment']
        total_weight = context['current_prec']
        threshold = context['threshold']

        next_base_params = otu.tree_add(base_updates, state.base_params)

        def get_offset_update(o_i, p_i, w_i, s_i):
            orig_shape = o_i.shape
            prec_axes = get_prec_axes(p_i, threshold)
            p_i_v = make_matrix(p_i, prec_axes)

            o_i_v = make_matrix(o_i, prec_axes)
            u_i_v = jnp.linalg.lstsq(s_i, w_i @ (p_i_v - o_i_v))[0]
            # u_i_v = jnp.linalg.solve(s_i, w_i @ (p_i_v -o_i_v))

            u_i = restore_shape(u_i_v, prec_axes, orig_shape)
            return u_i

        offset_update = jtu.tree_map(
            get_offset_update,
            state.offset,
            params,
            weight,
            total_weight
        )

        update = otu.tree_add(base_updates, offset_update)

        next_offset = otu.tree_add(state.offset, offset_update)

        next_state = MatrixAverageOffsetState(
            base_state=base_state,
            base_params=next_base_params,
            offset=next_offset,
        )

        return update, next_state

    return OnlineLearner(init_fn, update_fn)


