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



def get_prec_axes(p, threshold):
    prec_dims = []

    shape = p.shape
    best_axes = []
    best_prod = 1
    # iterate over all bit masks from 0 to 2^(len(shape)-1)
    for idx in range(2**(len(shape))):
        # get bit representation of idx:
        A = idx
        shape_prod = 1
        axes = []
        for s in range(len(shape)):
            if A % 2 == 1:
                shape_prod *= shape[s]
                axes.append(s)
            A = A // 2

        if shape_prod >= best_prod and shape_prod <= threshold:
            best_prod = shape_prod
            best_axes = axes
    return best_axes

def multiply(l):
    x = 1
    for f in l:
        x = x*f
    return x

def make_matrix(p, conditioned_axes):
    unconditioned_axes = [s for s in range(len(p.shape)) if s not in conditioned_axes]

    all_axes = conditioned_axes + unconditioned_axes


    first_dim = multiply([p.shape[s] for s in conditioned_axes])
    second_dim =multiply([p.shape[s] for s in unconditioned_axes])

    # bring axes we are going to precondition to the front
    p = jnp.transpose(p, all_axes)

    # flatten into a matrix
    p = jnp.reshape(p, (first_dim, second_dim))

    return p

def restore_shape(p, conditioned_axes, orig_shape):
    unconditioned_axes = [s for s in range(len(orig_shape)) if s not in conditioned_axes]

    all_axes = conditioned_axes + unconditioned_axes

    # need to invert this permutation
    inverse = [0] * len(all_axes)
    for idx, v in enumerate(all_axes):
        inverse[v] = idx


    current_shape = [orig_shape[s] for s in all_axes]


    p = jnp.reshape(p, current_shape)
    p = jnp.transpose(p, inverse)

    return p


# def get_prec_dim(p, threshold):
#     if p.size  <= threshold:
#         return 'full' # = flatten it  and do full preconditioning

#     # find the  largest dimension of size less than threshold.
#     best_index = 'diag' #=just do l2?
#     best_size = 0
#     for idx, size in enumerate(p.shape):
#         if size <= threshold and size >= best_size:
#             best_index = idx
#             best_size = size

#     return best_index

class PerParamState(NamedTuple):
    max_grad_norm: optax.Updates
    est_prec: optax.Updates
    # identity_scaling: optax.Updates
    sum_squared_grad: optax.Updates
    grad_sum: optax.Updates
    # weight_sum: optax.Updates
    # prec_dim: optax.Updates
    # d: optax.Updates

class ApproxSqrtDirectionState(NamedTuple):
    weight_sum: optax.Updates
    direction: optax.Updates
    per_param_state: PerParamState
    dual_sum: optax.Updates

def approx_sqrt_direction_full(
    eps: float=1.0,
    solver: str='solve',
    eps_type: str='trace',
    normalize:  bool=False,
    threshold: int=4096,
):

    def init_fn(params: optax.Params):

        def make_per_param_state(p):
            prec_axes = get_prec_axes(p, threshold)
            d = multiply([p.shape[i]  for i in prec_axes])
            p_v = make_matrix(p, prec_axes)
                        
            return PerParamState(
                max_grad_norm=jnp.zeros([]),
                est_prec=jnp.zeros((d,d)),
                sum_squared_grad=jnp.zeros([]),
                grad_sum=jnp.zeros_like(p_v),
                # prec_dim=prec_dim,
                # d=d,
            )

        def make_dual_sum(p):
            prec_axes = get_prec_axes(p, threshold)
            d = multiply([p.shape[i]  for i in prec_axes])
            return make_matrix(jnp.zeros_like(p), prec_axes)
            
        return ApproxSqrtDirectionState(
            weight_sum=jnp.zeros([]),
            direction=jax.tree.map(jnp.zeros_like,params),
            per_param_state=jax.tree.map(make_per_param_state,params),
            dual_sum=jax.tree.map(make_dual_sum, params)
        )


    def get_next_per_param_state(
            grad,
            state,
            next_weight_ratio,
    ):
        orig_shape = grad.shape

        prec_axes = get_prec_axes(grad, threshold)
        grad_v = make_matrix(grad, prec_axes)


        norm_grad = jnp.linalg.norm(grad) + 1e-8
        next_max_grad_norm = jnp.maximum(state.max_grad_norm*next_weight_ratio, norm_grad)
        next_grad_sum = (state.grad_sum + grad_v)*next_weight_ratio

        next_sum_squared_grad = (state.sum_squared_grad + norm_grad**2) * next_weight_ratio**2
        denom = jnp.maximum(jnp.sum(grad_v  * (state.est_prec @ grad_v))/norm_grad**2, norm_grad)

        denom = denom
        # jax.debug.print("denom: {d}",d=denom)

        next_est_prec = (state.est_prec + grad_v @ grad_v.T/denom) * next_weight_ratio


        return PerParamState(
            max_grad_norm=next_max_grad_norm,
            est_prec=next_est_prec,
            sum_squared_grad=next_sum_squared_grad,
            grad_sum=next_grad_sum,
            # prec_dim=state.prec_dim,
            # d=state.d
        )

    def get_regularized_prec(state, next_weight_sum):
        d = state.est_prec.shape[0]
        if eps_type == 'trace':
            eps_scale = jnp.linalg.trace(state.est_prec) / (d * next_weight_sum)
        elif eps_type == 'const':
            eps_scale = 1.0
        elif eps_type == 'norm':
            eps_scale = state.max_grad_norm/d
        else:
            raise ValueError(f'unknown eps type: {eps_type}')

        scaled_eps = eps * eps_scale + 1e-8 #jnp.finfo(state.est_prec.dtype).eps
        # jax.debug.print("eps: {e}",e=scaled_eps)

        regularized_preconditioner = state.est_prec + jnp.eye(d) * scaled_eps
        return regularized_preconditioner

    def get_direction(
        grad,
        next_dual_sum,
        regularized_preconditioner,
        state,
    ):
        orig_shape = grad.shape

        prec_axes = get_prec_axes(grad, threshold)
        next_direction = -next_dual_sum
        if normalize:
            next_direction = next_direction * jnp.minimum(1.0, 1.0/jnp.sqrt(jnp.sum(next_direction * (regularized_preconditioner @ next_direction))+1e-8))
        
        # next_direction = next_dual_sum * jnp.minimum(1.0, 1.0/jnp.sqrt(jnp.sum(state.grad_sum * next_dual_sum)+1e-8))
        next_direction = restore_shape(next_direction, prec_axes, orig_shape)

        return next_direction

    def get_next_dual_sum(
        grad,
        dual_sum,
        regularized_preconditioner,
        state,
        next_weight_sum
    ):
        orig_shape = grad.shape

        prec_axes = get_prec_axes(grad, threshold)
        grad_v = make_matrix(grad, prec_axes)
        d = multiply([orig_shape[s] for s in prec_axes])


        # # jax.debug.print("prec: {p}",p=jnp.linalg.trace(state.est_prec))
        # if eps_type == 'trace':
        #     eps_scale = jnp.linalg.trace(state.est_prec) / (d * next_weight_sum)
        # elif eps_type == 'const':
        #     eps_scale = 1.0
        # elif eps_type == 'norm':
        #     eps_scale = state.max_grad_norm/d
        # else:
        #     raise ValueError(f'unknown eps type: {eps_type}')

        # scaled_eps = eps * eps_scale
        # # jax.debug.print("eps: {e}",e=scaled_eps)

        # regularized_preconditioner = state.est_prec + jnp.eye(d) * scaled_eps

        if solver == 'svd':
            next_dual_sum = state.grad_sum
            u,s ,vh = jnp.linalg.svd(regularized_preconditioner)
            next_dual_sum = (u / s) @ vh @ state.grad_sum
        elif solver == 'lstsq':
            next_dual_sum = jnp.linalg.lstsq(regularized_preconditioner, state.grad_sum)[0]
        elif solver == 'solve':
            next_dual_sum = jnp.linalg.solve(regularized_preconditioner, state.grad_sum)
        elif solver == 'solve_inc':
            target = state.grad_sum - regularized_preconditioner @ dual_sum
            correction = jnp.linalg.solve(regularized_preconditioner, target)
            next_dual_sum = dual_sum + correction
        elif solver == 'cholesky':
            L = jax.lax.linalg.cholesky(regularized_preconditioner)
            intermediate = jax.lax.linalg.triangular_solve(
                L,
                state.grad_sum,
                left_side=True,
                lower=True,
                transpose_a=True
            )
            next_dual_sum = jax.lax.linalg.triangular_solve(
                L,
                intermediate,
                left_side=True,
                lower=True,
                transpose_a=False
            )
        elif solver == 'conjgrad1':
            # take one step of conj gradient descent from previous solution
            r = regularized_preconditioner @ dual_sum - state.grad_sum # [d, N]

            # compute optimal stepsize
            rTr = jnp.sum(r*r, axis=0) # [N]
            rTAr = 1e-8 + jnp.sum(r * (regularized_preconditioner @ r), axis=0) #[N]
            alpha = rTr/rTAr

            # jax.debug.print("alpha: {a}", a=alpha)
            
            next_dual_sum = dual_sum - r * alpha
        elif solver == 'solve_conjgrad1':
            next_dual_sum = jnp.linalg.solve(regularized_preconditioner, state.grad_sum)
            # take one step of conj gradient descent from this solution
            r = regularized_preconditioner @ next_dual_sum - state.grad_sum # [d, N]

            # compute optimal stepsize
            rTr = jnp.sum(r*r, axis=0) # [N]
            rTAr = 1e-8 + jnp.sum(r * (regularized_preconditioner @ r), axis=0) #[N]
            alpha = rTr/rTAr

            # jax.debug.print("alpha: {a}", a=alpha)
            
            next_dual_sum = next_dual_sum - r * alpha
            
            
            

        # next_dual_sum = next_dual_sum * jnp.minimum(1.0, 1.0 / jnp.sqrt(jnp.sum(next_dual_sum * (regularized_preconditioner @ next_dual_sum) + 1e-8)))
        # next_dual_sum = next_dual_sum * jnp.minimum(1.0, 1.0/jnp.sqrt(jnp.sum(state.grad_sum * next_dual_sum)+1e-8))

        # if normalize:
        #     next_dual_sum = next_dual_sum / (state.grad_sum.T @ next_dual_sum)



        # next_dual_sum = restore_shape(next_dual_sum, prec_axes, orig_shape)
        return next_dual_sum

        # return -next_dual_sum
    


    def update_fn(grads, state, next_weight_ratio, params, context=None):


        next_per_param_state = jax.tree.map(
            jax.tree_util.Partial(get_next_per_param_state, next_weight_ratio=next_weight_ratio),
            grads,
            state.per_param_state
        )
        next_weight_sum = (state.weight_sum +  1.0)*next_weight_ratio

        # prev_direction = jax.tree.map(
        #     get_direction,
        #     grads,
        #     state.dual_sum,
        #     state.per_param_state
        # )
        regularized_preconditioner = jax.tree.map(
            lambda s: get_regularized_prec(s, next_weight_sum),
            next_per_param_state,
            is_leaf=lambda s: isinstance(s, PerParamState)
        )

        next_dual_sum = jax.tree.map(
            jax.tree_util.Partial(get_next_dual_sum, next_weight_sum=next_weight_sum),
            grads,
            state.dual_sum,
            regularized_preconditioner,
            next_per_param_state
        )

        next_direction = jax.tree.map(
            get_direction,
            grads,
            next_dual_sum,
            regularized_preconditioner,
            next_per_param_state
        )


        updates = jax.tree.map(
            lambda x,y: x-y,
            next_direction,
            state.direction
        )

        next_state = ApproxSqrtDirectionState(
            weight_sum=next_weight_sum,
            direction=next_direction,
            per_param_state=next_per_param_state,
            dual_sum=next_dual_sum,
        )

        if context is not None:
            current_prec = jax.tree.map(
                lambda s: get_regularized_prec(s, next_weight_sum),
                next_state.per_param_state,
                is_leaf=lambda x: isinstance(x, PerParamState)
            )
            prec_increment = jax.tree.map(
                lambda next_s, s: next_s.est_prec - s.est_prec * next_weight_ratio,
                next_state.per_param_state,
                state.per_param_state,
                is_leaf=lambda x: isinstance(x, PerParamState)
            )

            context['current_prec'] = current_prec
            context['prec_increment'] = prec_increment
            context['threshold'] = threshold


        return updates, next_state

    return OnlineLearner(init_fn, update_fn)
