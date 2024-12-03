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


class DownscaleEpsilonState(NamedTuple):
    origin_regret: jax.Array
    max_ratio: jax.Array


# this procedure is used to convert algos with sqrt(T) origin regert
# into ones with O(1)  origin-regret
def downscale_epsilon(
    base_params: optax.Updates,
    next_weight_ratio: jax.Array,
    origin_regret: jax.Array,
    max_regret: jax.Array,
    max_grad_norm_hint: jax.Array,
    scale_strategy: str = "loglog",
):
    if scale_strategy == "log":
        scale_fn = lambda r: 1.0 / (1.0 + r)
    elif scale_strategy == "loglog":
        scale_fn = lambda r: 1.0 / ((1.0 + r) * jnp.log(1 + 1.0 / r))
    elif scale_strategy == "constant":
        scale_fn = lambda r: 1.0 / ((1.0 + r) * jnp.log(1.0 + 1.0 / r) ** 2)
    else:
        raise ValueError(f"unknown scale_strategy: {scale_strategy}")

    if isinstance(max_grad_norm_hint, jax.Array) and max_grad_norm_hint.size == 1:
        max_inner_product = max_grad_norm_hint * optax.tree_utils.tree_l2_norm(
            base_params
        )
    else:
        max_inner_product_tree = jax.tree.map(
            lambda a, b: jnp.sum(jnp.abs(b) * jnp.abs(a)),
            base_params,
            max_grad_norm_hint,
        )
        max_inner_product = optax.tree_utils.tree_l2_norm(max_inner_product_tree)

    next_max_regret = jnp.maximum(max_regret, origin_regret + max_inner_product)

    scaling = scale_fn(next_max_regret)

    return scaling, next_max_regret


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


def bigphi(V, S, z, k, eps, alpha):
    return littlephi(V + z + k * jnp.abs(S), S, eps, alpha)


def d2_bigphi(V, S, z, k, eps, alpha):
    # I will assume that jax.grad is just as good as manually implementing
    # the derivative until someone tells me otherwise.

    def to_vmap(v_, s_, z_, k_):
        def to_diff(s):
            return bigphi(v_, s, z_, k_, eps, alpha)

        return jax.grad(to_diff)(s_)

    result = jax.vmap(to_vmap)(V, S, z, k)
    jax.debug.print(
        "V: {v}\nS: {s}\nz: {z}\nk: {k}\neps\n{eps}\nalpha: {alpha}\nresult: {r}",
        v=V,
        s=S,
        r=result,
        z=z,
        k=k,
        eps=eps,
        alpha=alpha,
    )
    return result

    # return jax.grad(lambda s: bigphi(V, s, z, k, eps, alpha))(S)


#### I think this one is buggy....
class RefinedPDEPerCoordState(NamedTuple):
    V: jax.Array
    S: jax.Array
    h: jax.Array
    origin_regret: Optional[jax.Array]
    max_regret: Optional[jax.Array]
    scaling: jax.Array


def refined_pde_per_coord(
    eps: jax.Array = 1.0,
    G: jax.Array = 1e-8,
    alpha: jax.Array = 1.0,
    constant_origin_regret: bool = False,
):
    def init_fn(params: optax.Params):
        V = jax.tree.map(jnp.zeros_like, params)
        S = jax.tree.map(jnp.zeros_like, params)
        h = jax.tree.map(lambda p: jnp.full_like(p, fill_value=G), params)

        if constant_origin_regret:
            origin_regret = 0.0
            max_regret = 0.0
            scaling = eps
        else:
            origin_regret = None
            max_regret = None
            scaling = eps
        return RefinedPDEPerCoordState(
            V=V,
            S=S,
            h=h,
            origin_regret=origin_regret,
            max_regret=max_regret,
            scaling=scaling,
        )

    def update_fn(
        grads: optax.Updates,
        state: RefinedPDEPerCoordState,
        next_weight_ratio: jax.Array,
        params: optax.Params,
        context: Optional[Context] = None,
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

        wrapped_d2_bigphi = jax.tree_util.Partial(d2_bigphi, eps=eps, alpha=alpha)

        unscaled_prev_params = jax.tree_map(
            wrapped_d2_bigphi,
            V_t_minus_one,
            S_t_minus_one,
            z_t,
            k_t,
        )

        unscaled_next_params = jax.tree.map(
            wrapped_d2_bigphi,
            V_t,
            S_t,
            z_t_plus_one,
            k_t_plus_one,
        )

        if constant_origin_regret:
            next_origin_regret = (
                state.origin_regret
                + optax.tree_utils.tree_vdot(grads, unscaled_prev_params)
            ) * next_weight_ratio
            next_scaling, next_max_regret = downscale_epsilon(
                unscaled_next_params,
                next_weight_ratio,
                state.origin_regret,
                state.max_regret,
                h_t_plus_one,
            )
        else:
            next_origin_regret = None
            next_max_regret = None
            next_scaling = state.scaling

        updates = jax.tree_map(
            lambda n_, p_: next_scaling * n_ - state.scaling * p_,
            unscaled_next_params,
            unscaled_prev_params,
        )

        next_state = RefinedPDEPerCoordState(
            V=V_t,
            S=S_t,
            h=h_t_plus_one,
            origin_regret=next_origin_regret,
            max_regret=next_max_regret,
            scaling=next_scaling,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


class MirrorDescentState(NamedTuple):
    sum_squared_grad: jax.Array
    sum_squared_grad_over_max: jax.Array
    max_grad: jax.Array
    sum_grad: jax.Array
    param: jax.Array


def mirror_descent(eps: float = 1.0, p: float = 0.5):
    def init_fn(params: optax.Params):
        return MirrorDescentState(
            sum_squared_grad=jax.tree.map(jnp.zeros_like, params),
            max_grad=jax.tree.map(jnp.zeros_like, params),
            sum_grad=jax.tree.map(jnp.zeros_like, params),
            sum_squared_grad_over_max=jax.tree.map(jnp.zeros_like, params),
            param=jax.tree.map(jnp.copy, params),
        )

    def update_fn(
        grads: optax.Updates,
        state: MirrorDescentState,
        next_weight_ratio: jax.Array,
        param: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):

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
                (s_alpha - prev_s_alpha)/ prev_s_alpha,
                jnp.zeros_like(s_alpha),
            )

            exp_theta_diff = jnp.exp(theta - prev_theta)

            exp_theta_diff_minus_one = jnp.expm1(theta - prev_theta)

            updates = (
                s_alpha_ratio * exp_theta_diff_minus_one + s_alpha_ratio_minus_one
            ) * prev_w - s_alpha * exp_theta_diff_minus_one

            # updates = (exp_theta_diff * s_alpha_ratio - 1.0) * prev_w - s_alpha * (
            #     exp_theta_diff_minus_one
            # )

            # next_w = -s_alpha * alpha * (jnp.exp(theta) - 1.0)
            # updates = next_w - prev_w

            return updates

        def get_theta(sum_squared_grad, max_grad, sum_grad):
            k = 3.0

            switch = jnp.abs(sum_grad) <= 2 * k * sum_squared_grad / max_grad
            theta = jax.lax.select(
                switch,
                (sum_grad) ** 2 / (4 * k**2 * sum_squared_grad + 1e-8),
                jnp.abs(sum_grad) / (k * max_grad + 1e-8)
                - sum_squared_grad / (max_grad + 1e-8),
            )
            return theta

        def get_alpha(sum_squared_grad_over_max):
            if p == 0.5:
                c = 3.0
                alpha = jax.tree.map(
                    lambda m: eps / (jnp.sqrt(c + m) * jnp.log(c + m) ** 2),
                    sum_squared_grad_over_max,
                )
            else:
                c = 1.0
                alpha = jax.tree.map(
                    lambda m: eps / (c + m) ** p,
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

        next_param = optax.tree_utils.tree_add(state.param, updates)

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

        next_state = MirrorDescentState(
            sum_squared_grad_over_max=next_sum_squared_grad_over_max,
            max_grad=next_max_grad,
            sum_squared_grad=next_sum_squared_grad,
            sum_grad=next_sum_grad,
            param=next_param,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


class RecursiveOptState(NamedTuple):
    wealth: jax.Array
    bet_fraction: jax.Array
    base_state: optax.OptState
    max_grad: jax.Array


def recursive_optimizer(
    base_opt: OnlineLearner,
    eps: jax.Array = 1.0,
    max_bet_fraction_ratio: float = 0.5,
):
    #### CAUTION: ONLY WORKS WITH UNIFORM WEIGHTING!!! ####
    def init_fn(params: optax.Params):
        base_state = base_opt.init(params)
        bet_fraction = jax.tree.map(jnp.zeros_like, params)
        max_grad = 0.0
        reward = 0.0
        return RecursiveOptState(
            wealth=eps,
            bet_fraction=bet_fraction,
            base_state=base_state,
            max_grad=max_grad,
        )

    def update_fn(
        grads: optax.Updates,
        state: RecursiveOptState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        # max_grad is max_{i<t} |g_i| * w_i/w_t
        # where w_i is the ith weight and next_weight_ratio is r=w_t/w_{t+1}
        # so next max_grad is max(max_grad*r, |g_t|*r)

        # we want to clip the gradient so that |clipped_g_t|*w_t <= max_{i<t} |g_i| w_i
        # this is equivalent to |clipped_g_t| <= max_grad
        l2_norm_grads = optax.tree_utils.tree_l2_norm(grads)
        clip_ratio = jnp.minimum(state.max_grad / (l2_norm_grads + 1e-8), 1.0)
        clipped_grads = jax.tree.map(lambda g: clip_ratio * g, grads)

        next_max_grad = jnp.maximum(
            state.max_grad * next_weight_ratio, l2_norm_grads * next_weight_ratio
        )

        # wealth is (eps + sum_{i<t} <w_i * g_i, param_i>/w_t
        # so, next wealth is (wealth + <g_t, param_t>) * r

        # param_t is w_t * wealth_t * bet_t

        # so <g_t, param_t> = <g_t, bet_t> * wealth_t * w_t

        # wealth = eps + state.reward
        wealth = state.wealth

        def clip_bet_fraction(b, m):
            norm = optax.tree_utils.tree_l2_norm(b)
            return optax.tree_utils.tree_scalar_mul(
                jnp.minimum(max_bet_fraction_ratio / (m * norm + 1e-8), 1.0), b
            )

        clipped_bet_fraction = clip_bet_fraction(state.bet_fraction, state.max_grad)

        current_iterate = jax.tree.map(lambda b: b * wealth, clipped_bet_fraction)

        grad_dot_bet = optax.tree_utils.tree_vdot(clipped_bet_fraction, clipped_grads)

        next_wealth = (state.wealth + wealth * grad_dot_bet) * next_weight_ratio

        grad_log_loss = jax.tree.map(lambda g: g / (1.0 - grad_dot_bet), clipped_grads)

        def correct_grad_log_loss(g, b):
            b_norm = optax.tree_utils.tree_l2_norm(b)
            g_b_dot = optax.tree_utils.tree_vdot(g, b)
            do_correct = (g_b_dot >= 0) * (
                b_norm > (max_bet_fraction_ratio / state.max_grad)
            )
            correction = optax.tree_utils.tree_scalar_mul(
                do_correct * g_b_dot / (b_norm + 1e-8) ** 2, b
            )
            return optax.tree_utils.tree_sub(g, correction)

        grad_log_loss = correct_grad_log_loss(grad_log_loss, state.bet_fraction)

        bet_fraction_updates, next_base_state = base_opt.update(
            grad_log_loss,
            state.base_state,
            next_weight_ratio,
            params=state.bet_fraction,
            context=context,
        )

        next_bet_fraction = optax.apply_updates(
            state.bet_fraction, bet_fraction_updates
        )

        clipped_next_bet_fraction = clip_bet_fraction(next_bet_fraction, next_max_grad)

        next_iterate = optax.tree_utils.tree_scalar_mul(
            next_wealth, clipped_next_bet_fraction
        )

        updates = optax.tree_utils.tree_sub(next_iterate, current_iterate)

        next_state = RecursiveOptState(
            wealth=next_wealth,
            bet_fraction=next_bet_fraction,
            base_state=next_base_state,
            max_grad=next_max_grad,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


class ONSBettorState(NamedTuple):
    wealth: jax.Array
    bet_fraction: jax.Array
    max_grad: jax.Array
    bet_sum_squared_grad: jax.Array
    wealth_increment: jax.Array


def ons_bettor(
    eps: jax.Array = 1.0,
    max_bet_fraction: jax.Array = 0.5,
    do_wealth_increment: jax.Array = False,
):

    ons_eta = 2.0 / (2.0 - jnp.log(3))

    def init_fn(params: optax.Params):
        return ONSBettorState(
            wealth=eps,
            bet_fraction=optax.tree_utils.tree_zeros_like(params),
            max_grad=optax.tree_utils.tree_zeros_like(params),
            bet_sum_squared_grad=optax.tree_utils.tree_zeros_like(params),
            wealth_increment=0.0,
        )

    def update_fn(
        grads: optax.Updates,
        state: ONSBettorState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):

        abs_grad = jax.tree.map(jnp.abs, grads)

        next_wealth_increment = (state.wealth_increment + 1.0) * next_weight_ratio
        next_max_grad = jax.tree.map(
            lambda m, a: next_weight_ratio * jnp.maximum(m, a), state.max_grad, abs_grad
        )

        grads = jax.tree.map(
            lambda g, m, a: g * jnp.minimum(1.0, m / (a + 1e-8)),
            grads,
            state.max_grad,
            abs_grad,
        )

        bet_grad = jax.tree.map(lambda g, b: g / (1 - g * b), grads, state.bet_fraction)

        next_bet_sum_squared_grad = jax.tree.map(
            lambda s, g: (s + g**2) * next_weight_ratio,
            state.bet_sum_squared_grad,
            bet_grad,
        )

        next_bet_fraction = jax.tree.map(
            lambda b, s, g: b / next_weight_ratio
            - ons_eta * g * next_weight_ratio / (s + 1e-8),
            state.bet_fraction,
            next_bet_sum_squared_grad,
            bet_grad,
        )
        next_bet_fraction = jax.tree.map(
            lambda b, a: jnp.clip(
                b, -max_bet_fraction / (a + 1e-8), max_bet_fraction / (a + 1e-8)
            ),
            next_bet_fraction,
            next_max_grad,
        )

        next_wealth = jax.tree.map(
            lambda w, b, g: w * (1 - b * g) * next_weight_ratio,
            state.wealth,
            state.bet_fraction,
            grads,
        )
        next_wealth = next_wealth + do_wealth_increment * eps / next_wealth_increment

        prev_param = jax.tree.map(lambda w, b: w * b, state.wealth, state.bet_fraction)

        next_param = jax.tree.map(lambda w, b: w * b, next_wealth, next_bet_fraction)

        updates = optax.tree_utils.tree_sub(next_param, prev_param)

        next_state = ONSBettorState(
            wealth=next_wealth,
            max_grad=next_max_grad,
            bet_fraction=next_bet_fraction,
            bet_sum_squared_grad=next_bet_sum_squared_grad,
            wealth_increment=next_wealth_increment,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)


class ConstantBettorState(NamedTuple):
    wealth: jax.Array
    max_grad: jax.Array
    sum_squared_grad: jax.Array
    count: jax.Array


def constant_bettor(eps: jax.Array = 1.0, bet_fraction: jax.Array = 0.5):

    def init_fn(params: optax.Params):
        return ConstantBettorState(
            wealth=eps,
            max_grad=optax.tree_utils.tree_zeros_like(params),
            sum_squared_grad=optax.tree_utils.tree_zeros_like(params),
            count=jnp.zeros([]),
        )

    def update_fn(
        grads: optax.Updates,
        state: ONSBettorState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):

        abs_grad = jax.tree.map(jnp.abs, grads)

        next_count = (state.count + 1) * next_weight_ratio**2

        next_max_grad = jax.tree.map(
            lambda m, a: next_weight_ratio * jnp.maximum(m, a), state.max_grad, abs_grad
        )

        grads = jax.tree.map(
            lambda g, m, a: g * jnp.minimum(1.0, m / (a + 1e-8)),
            grads,
            state.max_grad,
            abs_grad,
        )

        next_sum_squared_grad = jax.tree.map(
            lambda s, g: (s + g**2) * next_weight_ratio, state.sum_squared_grad, grads
        )

        # scaled_next_bet_fraction = jax.tree_map(
        #     lambda s: bet_fraction/jnp.sqrt(s/(next_count+1e-8)+1e-8),
        #     next_sum_squared_grad
        # )

        # scaled_prev_bet_fraction = jax.tree_map(
        #     lambda s: bet_fraction/jnp.sqrt(s/(state.count+1e-8)+1e-8),
        #     state.sum_squared_grad
        # )

        scaled_next_bet_fraction = jax.tree_map(
            lambda s: bet_fraction / (s + 1e-8),
            next_max_grad,
        )

        scaled_prev_bet_fraction = jax.tree_map(
            lambda s: bet_fraction / (s + 1e-8),
            state.max_grad,
        )
        next_wealth = jax.tree.map(
            lambda w, b, g: w * (1 - b * g) * next_weight_ratio,
            state.wealth,
            scaled_next_bet_fraction,
            grads,
        )

        prev_param = jax.tree.map(
            lambda w, b: w * b,
            state.wealth,
            scaled_prev_bet_fraction,
        )

        next_param = jax.tree.map(
            lambda w, b: w * b, next_wealth, scaled_next_bet_fraction
        )

        updates = optax.tree_utils.tree_sub(next_param, prev_param)

        next_state = ConstantBettorState(
            wealth=next_wealth,
            max_grad=next_max_grad,
            sum_squared_grad=next_sum_squared_grad,
            count=next_count,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)
