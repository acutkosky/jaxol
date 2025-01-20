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
from jaxol.online_learner import (
    OnlineLearner,
    Context,
    to_OL,
    get_next_weight_ratio,
    get_next_averaging_factor,
    get_next_accumulation,
)


class LayerwiseMirrorDescentState(NamedTuple):
    sum_squared_grad: jax.Array
    sum_squared_grad_over_max: jax.Array
    max_grad: jax.Array
    sum_grad: jax.Array
    param: jax.Array


def layerwise_mirror_descent(eps: float = 1.0, p: float = 0.5, k=3.0, scale_eps=False):
    def init_fn(params: optax.Params):
        return LayerwiseMirrorDescentState(
            sum_squared_grad=jax.tree.map(lambda x: 0.0, params),
            max_grad=jax.tree.map(lambda x: 0.0, params),
            sum_grad=jax.tree.map(lambda x: 0.0, params),
            sum_squared_grad_over_max=jax.tree.map(lambda x: 0.0, params),
            param=jax.tree.map(lambda x: 0.0, params),
        )

    def update_fn(
        grads: optax.Updates,
        state: LayerwiseMirrorDescentState,
        next_weight_ratio: jax.Array,
        param: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):

        grads = jax.tree.map(jnp.sum, grads)

        abs_grad = jax.tree.map(jnp.abs, grads)
        next_max_grad = jax.tree.map(
            lambda m, a: jnp.maximum(m, a) * next_weight_ratio, state.max_grad, abs_grad
        )

        next_sum_squared_grad = jax.tree.map(
            lambda s, g: (s + g**2) * next_weight_ratio**2,
            state.sum_squared_grad,
            grads,
        )

        next_sum_squared_grad_over_max = jax.tree.map(
            lambda s, m, g: s + g**2 * next_weight_ratio**2 / (m**2 + 1e-8),
            state.sum_squared_grad_over_max,
            next_max_grad,
            grads,
        )

        next_sum_grad = jax.tree.map(
            lambda s, g: (s + g) * next_weight_ratio, state.sum_grad, grads
        )

        def get_updates(
            prev_w, sum_grad, prev_sum_grad, theta, prev_theta, alpha, prev_alpha
        ):
            s_alpha = jnp.sign(sum_grad) * alpha
            prev_s_alpha = jnp.sign(prev_sum_grad) * prev_alpha

            s_alpha_ratio = jax.lax.select(
                prev_s_alpha != 0.0,
                s_alpha / prev_s_alpha,
                jnp.ones_like(s_alpha),
            )

            s_alpha_ratio_minus_one = jax.lax.select(
                prev_s_alpha != 0.0,
                s_alpha / prev_s_alpha - 1.0,  # (s_alpha - prev_s_alpha)/ prev_s_alpha,
                jnp.zeros_like(s_alpha),
            )

            exp_theta_diff = jnp.exp(theta - prev_theta)

            exp_theta_diff_minus_one = jnp.expm1(theta - prev_theta)

            # updates = (
            #     s_alpha_ratio * exp_theta_diff_minus_one + s_alpha_ratio_minus_one
            # ) * prev_w - s_alpha * exp_theta_diff_minus_one

            updates = (exp_theta_diff * s_alpha_ratio - 1.0) * prev_w - s_alpha * (
                exp_theta_diff_minus_one
            )

            # next_w = -s_alpha * alpha * (jnp.exp(theta) - 1.0)
            # updates = next_w - prev_w

            return updates

        def get_theta(sum_squared_grad, max_grad, sum_grad):
            # k is set as argument in the original learner construction.

            switch = jnp.abs(sum_grad) <= 2 * k * sum_squared_grad / max_grad
            theta = jax.lax.select(
                switch,
                (sum_grad) ** 2 / (4 * k**2 * sum_squared_grad + 1e-8),
                jnp.abs(sum_grad) / (k * max_grad + 1e-8)
                - sum_squared_grad / (max_grad + 1e-8),
            )
            return theta

        if scale_eps:
            rescaled_eps = eps/optax.tree_utils.tree_sum(jax.tree.map(lambda p: p.ndim, grads))
        else:
            rescaled_eps = eps
        def get_alpha(sum_squared_grad_over_max):
            if p == 0.5:
                c = 3.0
                alpha = jax.tree.map(
                    lambda m: rescaled_eps / (jnp.sqrt(c + m) * jnp.log(c + m) ** 2),
                    sum_squared_grad_over_max,
                )
            else:
                c = 1.0
                alpha = jax.tree.map(
                    lambda m: rescaled_eps / (c + m) ** p,
                    sum_squared_grad_over_max,
                )
            return alpha

        next_theta = jax.tree.map(
            get_theta, next_sum_squared_grad, next_max_grad, next_sum_grad
        )
        prev_theta = jax.tree.map(
            get_theta, state.sum_squared_grad, state.max_grad, state.sum_grad
        )
        next_alpha = jax.tree.map(get_alpha, next_sum_squared_grad_over_max)
        prev_alpha = jax.tree.map(get_alpha, state.sum_squared_grad_over_max)
        updates = jax.tree.map(
            get_updates,
            state.param,
            next_sum_grad,
            state.sum_grad,
            next_theta,
            prev_theta,
            next_alpha,
            prev_alpha,
        )

        next_param = jax.tree.map(
            lambda a,b: a+b,
            state.param,
            updates
        )

        # def get_param_value(
        #     sum_squared_grad_over_max, max_grad, sum_squared_grad, sum_grad
        # ):

        #     if p == 0.5:
        #         c = 3.0
        #         alpha = jax.tree.map(
        #             lambda m: eps / (jnp.sqrt(c + m) * jnp.log(c + m) ** 2),
        #             sum_squared_grad_over_max,
        #         )
        #     else:
        #         c = 1.0
        #         alpha = jax.tree.map(
        #             lambda m: eps / (c + m) ** p,
        #             sum_squared_grad_over_max,
        #         )

        #     k = 3.0

        #     switch = jnp.abs(sum_grad) <= 2 * k * sum_squared_grad / max_grad
        #     theta = jax.lax.select(
        #         switch,
        #         (sum_grad) ** 2 / (4 * k**2 * sum_squared_grad + 1e-8),
        #         jnp.abs(sum_grad) / (k * max_grad + 1e-8)
        #         - sum_squared_grad / (max_grad + 1e-8),
        #     )

        #     # jax.debug.print("theta: {t}\nsum_grad: {s}\nsum_squared_grad: {q}\nmax_grad: {m}\nswitch: {w}",t=theta,s=sum_grad,q=sum_squared_grad_over_max, m=max_grad, w=switch)

        #     result = -jnp.sign(sum_grad) * alpha * (jnp.exp(theta) - 1.0)

        #     return result

        # prev_param = get_param_value(
        #     state.sum_squared_grad_over_max,
        #     state.max_grad,
        #     state.sum_squared_grad,
        #     state.sum_grad,
        # )

        # next_param = get_param_value(
        #     next_sum_squared_grad_over_max,
        #     next_max_grad,
        #     next_sum_squared_grad,
        #     next_sum_grad,
        # )

        # updates = optax.tree_utils.tree_sub(next_param, prev_param)

        next_state = LayerwiseMirrorDescentState(
            sum_squared_grad_over_max=next_sum_squared_grad_over_max,
            max_grad=next_max_grad,
            sum_squared_grad=next_sum_squared_grad,
            sum_grad=next_sum_grad,
            param=next_param,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)

