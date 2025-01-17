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
    best_sum = 0
    # iterate over all bit masks from 0 to 2^(len(shape)-1)
    for idx in range(2**(len(shape))):
        print("idx: ",idx)
        # get bit representation of idx:
        A = idx
        shape_sum = 0
        axes = []
        for s in range(len(shape)):
            if A % 2 == 1:
                shape_sum += shape[s]
                axes.append(s)
            A = A // 2

        if shape_sum > best_sum and shape_sum < threshold:
            best_sum = shape_sum
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

    print("conditioned axes: ",conditioned_axes)
    print("all axes: ",all_axes)

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

    print("restore all axes: ",all_axes)
    print("invers: ",inverse)

    current_shape = [orig_shape[s] for s in all_axes]


    p = jnp.reshape(p, current_shape)
    p = jnp.transpose(p, inverse)

    return p


def get_prec_dim(p, threshold=4096):
    if p.size  <= threshold:
        return 'full' # = flatten it  and do full preconditioning

    # find the  largest dimension of size less than threshold.
    best_index = 'diag' #=just do l2?
    best_size = 0
    for idx, size in enumerate(p.shape):
        if size <= threshold and size >= best_size:
            best_index = idx
            best_size = size

    return best_index

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
            
        return ApproxSqrtDirectionState(
            weight_sum=jnp.zeros([]),
            direction=jax.tree.map(jnp.zeros_like,params),
            per_param_state=jax.tree.map(make_per_param_state,params)
        )


    def get_next_per_param_state(
            grad,
            state,
            next_weight_ratio,
    ):
        orig_shape = grad.shape

        prec_axes = get_prec_axes(grad, threshold)
        grad_v = make_matrix(grad, prec_axes)


        norm_grad = jnp.linalg.norm(grad)
        next_max_grad_norm = jnp.maximum(state.max_grad_norm*next_weight_ratio, norm_grad)
        next_grad_sum = (state.grad_sum + grad_v)*next_weight_ratio

        next_sum_squared_grad = (state.sum_squared_grad + norm_grad**2) * next_weight_ratio**2
        denom = jnp.maximum(jnp.sum(grad_v  * (state.est_prec @ grad_v))/norm_grad**2, norm_grad)

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

    def get_next_direction(
        grad,
        state,
        next_weight_sum
    ):
        orig_shape = grad.shape

        prec_axes = get_prec_axes(grad, threshold)
        grad_v = make_matrix(grad, prec_axes)
        d = multiply([orig_shape[s] for s in prec_axes])

        # jax.debug.print("prec: {p}",p=jnp.linalg.trace(state.est_prec))
        if eps_type == 'trace':
            eps_scale = jnp.linalg.trace(state.est_prec) / (d * next_weight_sum)
        elif eps_type == 'const':
            eps_scale = 1.0
        elif eps_type == 'norm':
            eps_scale = state.max_grad_norm/d
        else:
            raise ValueError(f'unknown eps type: {eps_type}')

        scaled_eps = eps * eps_scale
        # jax.debug.print("eps: {e}",e=scaled_eps)

        regularized_preconditioner = state.est_prec + jnp.eye(d) * scaled_eps

        if solver == 'solve':
            next_dual_sum = jnp.linalg.solve(regularized_preconditioner, state.grad_sum)

        next_dual_sum = next_dual_sum * jnp.minimum(1.0, 1.0/(jnp.sum(state.grad_sum * next_dual_sum)))

        # if normalize:
        #     next_dual_sum = next_dual_sum / (state.grad_sum.T @ next_dual_sum)

        next_dual_sum = restore_shape(next_dual_sum, prec_axes, orig_shape)

        return -next_dual_sum
    


    def update_fn(grads, state, next_weight_ratio, params, context=None):


        next_per_param_state = jax.tree.map(
            jax.tree_util.Partial(get_next_per_param_state, next_weight_ratio=next_weight_ratio),
            grads,
            state.per_param_state
        )
        next_weight_sum = (state.weight_sum +  1.0)*next_weight_ratio

        next_direction = jax.tree.map(
            jax.tree_util.Partial(get_next_direction, next_weight_sum=next_weight_sum),
            grads,
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
            per_param_state=next_per_param_state
        )


        return updates, next_state

    return OnlineLearner(init_fn, update_fn)
