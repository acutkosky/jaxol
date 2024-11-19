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


# Define the "weight_ratio" r_t = w_{t-1}/w_t ,  where w_t is the weight on the t^th iterate.
# Define the "averaging_factor" alpha_t = w_t/w_{1:t}
# Define the "averaging_squared_factor" gamma_t = w_t^2/w^2_{1:t}

# r_t and alpha_t are related as follows:
# 1/alpha_t = 1/alpha_{t-1} * r_t  + 1
# alpha_t =  alpha_{t-1}/(r_t + alpha_{t-1})
# r_t = alpha_{t-1}(1/alpha_t - 1)

# r_t and gamma_t are related as follows:
# 1/gamma_t = 1/gamma_{t-1} * r_t^2 + 1
# gamma_t =  gamma_{t-1}/(r_t^2 + gamma_{t-1})
# r_t = sqrt(gamma_{t-1}(1/gamma_t - 1))


# Also if for some sequence x_t we define:
# S_T = sum_{t=1}^T w_t x_t / w_{T+1}
# then we have:
# S_T = (S_{T-1} + x_T)*r_{T+1}
# Also, if
# Z_T = sum_{t=1}^T w_t^p x_t/w_{T+1}^p
# then we have:
# Z_T = (Z_{t-1} + x_t)* r_{T+1}^p

# We will ask users to provide us with the next weight ratio r_{t+1} at time t.



def get_next_weight_ratio(averaging_factor, next_averaging_factor):
    '''computes weight ratio r_{t+1} from averaging factors alpha_t and alpha_{t+1}'''
    return averaging_factor * (1.0/ next_averaging_factor - 1.0)


def get_next_averaging_factor(next_weight_ratio, averaging_factor):
    '''computes averaging factor alpha_{t+1} from weight ratio r_{t+1} and alpha_t'''
    return averaging_factor / (next_weight_ratio + averaging_factor)


def get_next_accumulation(next_weight_ratio, accumulation, next_value):
    '''computes S_T  from r_{T+1}, S_{T-1}, and x_T'''
    return accumulation * next_weight_ratio + next_value


Context = NamedTuple


class OnlineLearner(NamedTuple):
    init: Callable[[optax.Params], optax.State]
    update: Callable[
        [
            optax.Updates,
            optax.State,
            jax.Array,
            Optional[Context],
            Optional[optax.Params],
        ],
        [optax.Updates, optax.State],
    ]


def GT_to_OL(tx: optax.GradientTransformation):
    def init_fn(params):
        return tx.init(params)

    def update_fn(
        grads: optax.Updates,
        state: NamedTuple,
        next_weight_ratio: jax.Array,
        context: Optional[Context] = None,
        params: Optional[optax.Params] = None,
    ):
        del context
        return tx.update(grads, state, params)

    return OnlineLearner(init_fn, update_fn)


class OLtoGTState(NamedTuple):
    ol_state: Any
    count: jax.Array


def OL_to_GT(ol: OnlineLearner):
    def init_fn(params):
        ol_state = ol.init(params)
        return OLtoGTState(ol_state=ol_state, count=0.0)

    def update_fn(
        grads,
        state,
        next_weight_ratio: Optional[jax.Array],
        context: Optional[Context] = None,
        params: Optional[optax.Params] = None,
    ):
        if next_weight_ratio is None:
            next_weight_ratio = 1.0

        ol_update, ol_state = ol.update(
            grads,
            state.ol_state,
            next_weight_ratio=next_weight_ratio,
            context=context,
            params=params,
            next_weight_ratio=next_weight_ratio,
        )
        return ol_update, ol_state

    return optax.GradientTransformation(ol.init, update_fn)
