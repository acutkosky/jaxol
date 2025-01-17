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



class UnitDirectionLearnerState(NamedTuple):
    grad_sum: optax.Updates
    s_sum: jax.Array
    prev_s_sum: jax.Array


def unit_direction_learner(preprocess_grads=lambda g: g) -> OnlineLearner:
    def init_fn(params):
        return UnitDirectionLearnerState(
            grad_sum=otu.tree_zeros_like(params),
            s_sum=1.0,
            prev_s_sum=0.0,  # jnp.zeros(1)
        )

    def update_fn(
        grads: optax.Updates,
        state: UnitDirectionLearnerState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        orig_grads = grads
        grads = preprocess_grads(grads)
        # jax.debug.print("grads: {g}",g=orig_grads)
        # jax.debug.print("grad norm: {n}, new norm: {p}",n=optax.tree_utils.tree_l2_norm(orig_grads), p=optax.tree_utils.tree_l2_norm(grads))
        next_grad_sum = jtu.tree_map(
            lambda s, g: (s + g)
            * next_weight_ratio,  # get_next_accumulation(next_weight_ratio, s, g),
            state.grad_sum,
            grads,
        )
        next_s = (
            otu.tree_vdot(state.grad_sum, grads)
            / (otu.tree_l2_norm(state.grad_sum) + 1e-8)
            * jnp.sign(state.s_sum)
        )

        next_s_sum = jtu.tree_map(
            lambda old_sum, s: (old_sum + s)
            * next_weight_ratio,  # get_next_accumulation(next_weight_ratio, old_sum, s),
            state.s_sum,
            next_s,
        )

        next_direction = otu.tree_scalar_mul(
            -jnp.sign(state.s_sum) / (otu.tree_l2_norm(next_grad_sum) + 1e-8),
            next_grad_sum,
        )
        # next_direction = jax.tree.map(lambda x: -x, next_direction)

        prev_direction = otu.tree_scalar_mul(
            -jnp.sign(state.prev_s_sum) / (otu.tree_l2_norm(state.grad_sum) + 1e-8),
            state.grad_sum,
        )

        updates = otu.tree_sub(next_direction, prev_direction)

        next_state = UnitDirectionLearnerState(
            grad_sum=next_grad_sum,
            s_sum=next_s_sum,
            prev_s_sum=state.s_sum,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)



class PreconditionedDirectionLearnerState(NamedTuple):
    grad_sum: optax.Updates
    grad_outer_sum: optax.Updates
    dual_sum: optax.Updates # g_{1:t-1}
    max_grad: optax.Updates
    s_sum: jax.Array # s_{1:t-1}
    prev_s_sum: jax.Array # s_{1:t-2}


def preconditioned_direction_learner(preprocess_grads=lambda g: g, epsilon=1e-8, root=True) -> OnlineLearner:
    def init_fn(params):
        return PreconditionedDirectionLearnerState(
            grad_sum=otu.tree_zeros_like(params),
            grad_outer_sum=jtu.tree_map(
                lambda p: jnp.outer(jnp.zeros_like(p).flatten(), jnp.zeros_like(p).flatten()),
                params
            ),
            # grad_outer_sum=jtu.tree_map(
            #     lambda p: jnp.eye(p.flatten().shape[0]) * epsilon,
            #     params
            # ),
            s_sum=0.0,
            dual_sum=otu.tree_zeros_like(params),
            max_grad=jtu.tree_map(
                lambda p: epsilon,
                params
            ),
            prev_s_sum=0.0,  # jnp.zeros(1)
        )

    def update_fn(
        grads: optax.Updates,
        state: UnitDirectionLearnerState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        orig_grads = grads
        grads = preprocess_grads(grads)
        # jax.debug.print("grads: {g}",g=orig_grads)
        # jax.debug.print("grad norm: {n}, new norm: {p}",n=optax.tree_utils.tree_l2_norm(orig_grads), p=optax.tree_utils.tree_l2_norm(grads))
        next_grad_sum = jtu.tree_map(
            lambda s, g: (s + g)
            * next_weight_ratio,  # get_next_accumulation(next_weight_ratio, s, g),
            state.grad_sum,
            grads,
        )
        flat_grads = jtu.tree_map(
            lambda g: g.reshape((-1,1)),
            grads
        )
        next_grad_outer_sum = jtu.tree_map(
            lambda s, g: (s + g @ g.T) * next_weight_ratio**2,
            state.grad_outer_sum,
            flat_grads
        )
        # next_grad_outer_sum = jtu.tree_map(
        #     lambda s, g: (s + jnp.outer(g.flatten(), g.flatten())) * next_weight_ratio**2,
        #     state.grad_outer_sum,
        #     grads
        # )
        next_max_grad = jtu.tree_map(
            lambda s, g: jnp.maximum(jnp.linalg.norm(g), s*next_weight_ratio),
            state.max_grad,
            grads
        )

        def matrix_prec(m):
            # return jnp.eye(m.shape[0])

            if root:
                # For some reason, specifying hermitian=True to the SVD call below
                # makes it not work too well. I have no clue why.
                u, s, vh = jnp.linalg.svd(m)
                prec_s =  1.0/jnp.sqrt(s + epsilon) #next_max_grad**2)
                # prec_s =  1.0/jnp.sqrt(s) #next_max_grad**2)
                return (u  * prec_s) @ vh
            else:
                return jnp.linalg.pinv(m)
        
        inv_sqrt_outer_sum = jtu.tree_map(
            lambda s: matrix_prec(s),
            next_grad_outer_sum
        )

        def inner_prod(a, m, b):
            # return jnp.sum(a*b)
            return a.T @ m @ b
        # def inner_prod(a, m, b):
        #     # return jnp.sum(a*b)
        #     return (a.flatten().reshape(1, -1) @ m @ b.flatten().reshape(-1,1)).flatten()
        def tree_inner_prod(t1, mat, t2):
            return otu.tree_sum(jtu.tree_map(
                lambda a,b,m: inner_prod(a, m, b),
                t1,
                t2,
                mat
            ))

        def tree_mat_norm(t):
            return jnp.sqrt(tree_inner_prod(t,inv_sqrt_outer_sum, t))
        def dual_vector(t, m):
            flat_t = t.reshape((-1,1))
            return (m @ flat_t / jnp.sqrt(1e-8 + inner_prod(flat_t, m ,flat_t))).reshape(t.shape)

        # def dual_vector(t, m):
        #     return (m @ t.flatten() / jnp.sqrt(1e-8 + inner_prod(t, m ,t))).reshape(t.shape)
            
        
        # next_s = (
        #     tree_inner_prod(state.dual_sum, inv_sqrt_outer_sum, grads)  * jnp.sign(state.s_sum)
        # )

        def sign(x):
            return jnp.where(x>=0, 1.0, -1.0)
        
        prev_direction = otu.tree_scalar_mul(
            sign(state.s_sum), state.dual_sum
        )

        next_s = (
            otu.tree_vdot(prev_direction, grads)
        ) # s_t

        next_dual_sum = jtu.tree_map(
            dual_vector,
            next_grad_sum,
            inv_sqrt_outer_sum
        )

        next_s_sum = jtu.tree_map(
            lambda old_sum, s: (old_sum + s)
            * next_weight_ratio,  # get_next_accumulation(next_weight_ratio, old_sum, s),
            state.s_sum,
            next_s,
        )

        next_direction = otu.tree_scalar_mul(
            sign(next_s_sum),
            next_dual_sum
        )
        # next_direction = otu.tree_scalar_mul(-1, next_dual_sum)
        # next_direction = jax.tree.map(lambda x: -x, next_direction)

        # prev_direction = otu.tree_scalar_mul(
        #     -jnp.sign(state.prev_s_sum), state.dual_sum
        # )
        # prev_direction = otu.tree_scalar_mul(-1, state.dual_sum)

        updates = otu.tree_sub(next_direction, prev_direction)

        # jax.debug.print("s: {s}, dir norm: {d}", s=next_s, d = jnp.linalg.norm(next_direction))

        next_state = PreconditionedDirectionLearnerState(
            grad_sum=next_grad_sum,
            grad_outer_sum=next_grad_outer_sum,
            max_grad=next_max_grad,
            s_sum=next_s_sum,
            dual_sum=next_dual_sum,
            prev_s_sum=state.s_sum,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)



class ApproxRootDirectionLearnerState(NamedTuple):
    grad_sum: optax.Updates
    inv_prec: optax.Updates
    dual_sum: optax.Updates # g_{1:t-1}
    max_grad: optax.Updates
    s_sum: jax.Array # s_{1:t-1}
    prev_s_sum: jax.Array # s_{1:t-2}


def approxroot_direction_learner(preprocess_grads=lambda g: g, epsilon=1e-8) -> OnlineLearner:
    def init_fn(params):
        return ApproxRootDirectionLearnerState(
            grad_sum=otu.tree_zeros_like(params),
            inv_prec=jtu.tree_map(
                lambda p: jnp.outer(jnp.zeros_like(p).flatten(), jnp.zeros_like(p).flatten()),
                params
            ),
            s_sum=0.0,
            dual_sum=otu.tree_zeros_like(params),
            max_grad=jtu.tree_map(
                lambda p: epsilon,
                params
            ),
            prev_s_sum=0.0,  # jnp.zeros(1)
        )

    def update_fn(
        grads: optax.Updates,
        state: UnitDirectionLearnerState,
        next_weight_ratio: jax.Array,
        params: Optional[optax.Params] = None,
        context: Optional[Context] = None,
    ):
        orig_grads = grads
        grads = preprocess_grads(grads)
        # jax.debug.print("grads: {g}",g=orig_grads)
        # jax.debug.print("grad norm: {n}, new norm: {p}",n=optax.tree_utils.tree_l2_norm(orig_grads), p=optax.tree_utils.tree_l2_norm(grads))
        next_grad_sum = jtu.tree_map(
            lambda s, g: (s + g)
            * next_weight_ratio,  # get_next_accumulation(next_weight_ratio, s, g),
            state.grad_sum,
            grads,
        )
        def inv_prec_denom(m, g):
            # u, s, vh = jnp.linalg.svd(m + jnp.outer(g.flatten(), g.flatten())/jnp.linalg.norm(g), hermitian=True)
            # prec_s =  1.0/(s + 1e-8) #next_max_grad**2)
            # jax.debug.print("s: {s}", s=s)
            # return (u*s) @vh
            return jnp.linalg.pinv(m + jnp.outer(g.flatten(), g.flatten())/ jnp.linalg.norm(g))

        def get_inv_prec(s, g):
            m = inv_prec_denom(s, g)
            g = g.flatten()
            g_outer = jnp.outer(g,g)
            g = g.reshape((-1,1))

            
            return (s + g_outer  /(jnp.sqrt(2)*jnp.maximum(jnp.linalg.norm(s @ g), jnp.linalg.norm(g))))*next_weight_ratio
            # return (s + g_outer @ m @ g_outer/(jnp.linalg.norm(g)**2))*next_weight_ratio
            # return (s  + m @ g_outer) *  next_weight_ratio
        next_inv_prec = jtu.tree_map(
            get_inv_prec,
            # lambda s, g: (s + ginv_prec_denom(s, g) @ jnp.outer(g.flatten(), g.flatten())) * next_weight_ratio,
            state.inv_prec,
            grads
        )
        # jax.debug.print("next inv prec: {x}",x=next_inv_prec)
        # next_grad_outer_sum = jtu.tree_map(
        #     lambda s, g: (s + jnp.outer(g.flatten(), g.flatten())) * next_weight_ratio**2,
        #     state.grad_outer_sum,
        #     grads
        # )
        next_max_grad = jtu.tree_map(
            lambda s, g: jnp.maximum(jnp.linalg.norm(g), s*next_weight_ratio),
            state.max_grad,
            grads
        )

        def matrix_prec(m, g):
            return jnp.linalg.pinv(m)
                
                
        
        inv_sqrt_outer_sum = jtu.tree_map(
            # lambda s, g: inv_prec_denom(s, g),
            lambda s, g: matrix_prec(s, g),
            next_inv_prec,
            grads
        )

        def inner_prod(a, m, b):
            # return jnp.sum(a*b)
            return (a.flatten().reshape(1, -1) @ m @ b.flatten().reshape(-1,1)).flatten()
        def tree_inner_prod(t1, mat, t2):
            return otu.tree_sum(jtu.tree_map(
                lambda a,b,m: inner_prod(a, m, b),
                t1,
                t2,
                mat
            ))

        def tree_mat_norm(t):
            return jnp.sqrt(tree_inner_prod(t,inv_sqrt_outer_sum, t))

        def dual_vector(t, m):
            return (m @ t.flatten() / jnp.sqrt(1e-8 + inner_prod(t, m ,t))).reshape(t.shape)
            
        
        # next_s = (
        #     tree_inner_prod(state.dual_sum, inv_sqrt_outer_sum, grads)  * jnp.sign(state.s_sum)
        # )

        def sign(x):
            return jnp.where(x>=0, 1.0, -1.0)
        
        prev_direction = otu.tree_scalar_mul(
            sign(state.s_sum), state.dual_sum
        )

        next_s = (
            otu.tree_vdot(prev_direction, grads)
        ) # s_t

        next_dual_sum = jtu.tree_map(
            dual_vector,
            next_grad_sum,
            inv_sqrt_outer_sum
        )

        next_s_sum = jtu.tree_map(
            lambda old_sum, s: (old_sum + s)
            * next_weight_ratio,  # get_next_accumulation(next_weight_ratio, old_sum, s),
            state.s_sum,
            next_s,
        )

        next_direction = otu.tree_scalar_mul(
            sign(next_s_sum),
            next_dual_sum
        )
        # next_direction = otu.tree_scalar_mul(-1, next_dual_sum)
        # next_direction = jax.tree.map(lambda x: -x, next_direction)

        # prev_direction = otu.tree_scalar_mul(
        #     -jnp.sign(state.prev_s_sum), state.dual_sum
        # )
        # prev_direction = otu.tree_scalar_mul(-1, state.dual_sum)

        updates = otu.tree_sub(next_direction, prev_direction)

        # jax.debug.print("s: {s}, dir norm: {d}", s=next_s, d = jnp.linalg.norm(next_direction))

        next_state = ApproxRootDirectionLearnerState(
            grad_sum=next_grad_sum,
            inv_prec=next_inv_prec,
            # grad_outer_sum=next_grad_outer_sum,
            max_grad=next_max_grad,
            s_sum=next_s_sum,
            dual_sum=next_dual_sum,
            prev_s_sum=state.s_sum,
        )

        return updates, next_state

    return OnlineLearner(init_fn, update_fn)

       
