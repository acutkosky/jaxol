import jax
from jax import numpy as jnp
import optax

def tree_l2_normalize(t, r=1.0):
    norm = optax.tree_utils.tree_l2_norm(t)+1e-8
    return jax.tree.map(
        lambda x: x*r/norm,
        t
    )

def tree_l2_clip(t, r=1.0):
    norm = optax.tree_utils.tree_l2_norm(t)+1e-8
    scale = jnp.minimum(1.0, r/norm)
    return jax.tree.map(
        lambda x: x*scale,
        t
    )
    
