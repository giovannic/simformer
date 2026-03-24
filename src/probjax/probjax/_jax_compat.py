# Compatibility shim for JAX 0.9+ where jax.core internals moved to jax._src.core
# and the "stackless" refactor removed sublevel/level machinery.
from contextlib import contextmanager

try:
    from jax.core import (
        Jaxpr, JaxprEqn, Literal, Var, Atom, ClosedJaxpr,
        Primitive, CallPrimitive, eval_jaxpr, ShapedArray,
    )
except ImportError:
    from jax._src.core import (
        Jaxpr, JaxprEqn, Literal, Var, Atom, ClosedJaxpr,
        Primitive, CallPrimitive, eval_jaxpr, ShapedArray,
    )

try:
    from jax.core import new_sublevel
except ImportError:
    # Removed in JAX 0.9 stackless refactor — no longer needed.
    @contextmanager
    def new_sublevel():
        yield

try:
    from jax import core
except ImportError:
    from jax._src import core

try:
    from jax import linear_util as lu
except ImportError:
    from jax._src import linear_util as lu

try:
    from jax.experimental.pjit import pjit_p
except ImportError:
    # Renamed to jit_p in JAX 0.9+
    from jax._src.pjit import jit_p as pjit_p

try:
    from jax.custom_derivatives import custom_jvp_call_p
except ImportError:
    from jax._src.custom_derivatives import custom_jvp_call_p

try:
    from jax._src.api_util import flatten_fun_nokwargs
except ImportError:
    flatten_fun_nokwargs = None

try:
    from jax._src.api_util import shaped_abstractify
except ImportError:
    from jax._src.core import shaped_abstractify

try:
    from jax.interpreters.batching import batch_jaxpr
except ImportError:
    from jax._src.interpreters.batching import batch_jaxpr

try:
    from jax.random import KeyArray
except ImportError:
    from jax import Array as KeyArray

try:
    from jax.random import PRNGKeyArray
except ImportError:
    PRNGKeyArray = KeyArray

try:
    from jax import tree_map, tree_flatten, tree_unflatten, tree_structure
except ImportError:
    from jax.tree_util import (
        tree_map,
        tree_flatten,
        tree_unflatten,
        tree_structure,
    )
