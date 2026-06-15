"""Defines functions for efficient computation of sparse Jacobians."""

from typing import Any, Callable, NamedTuple, NewType, Tuple, Union

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import networkx
import numpy as onp
import scipy.sparse as ssparse

# Coloring strategy employed to find structurally-independent output elements.
_DEFAULT_COLORING_STRATEGY = "largest_first"

ArrayWithOptionalAux = Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]

@jax.tree_util.register_static
class StaticInt(int):
    pass

class ReverseModeColoring(NamedTuple):
    """Precomputed coloring for use with `jacrev`.

    Attributes:
        sparsity: Sparse matrix identifying nonzero Jacobian locations.
        output_coloring: Integer array of shape `(nrows,)` assigning each
            output element (corresponding to a row in the Jacobian) a color
            index in `[0, ncolors)`.
        ncolors: Total number of colors used. Static, because it is used as
            part of an array shape.
    """

    sparsity: jsparse.BCOO
    output_coloring: jnp.ndarray
    ncolors: StaticInt


class ForwardModeColoring(NamedTuple):
    """Precomputed coloring for use with `jacfwd`.

    Attributes:
        sparsity: Sparse matrix identifying nonzero Jacobian locations.
        input_coloring: Integer array of shape `(ncols,)` assigning each
            input element (corresponding to a column in the Jacobian) a color
            index in `[0, ncolors)`.
        ncolors: Total number of colors used. Static, because it is used as
            part of an array shape.
    """

    sparsity: jsparse.BCOO
    input_coloring: jnp.ndarray
    ncolors: StaticInt


def jacrev_coloring(
    sparsity: Union[jsparse.BCOO, ssparse.spmatrix],
    coloring_strategy: str = _DEFAULT_COLORING_STRATEGY,
) -> ReverseModeColoring:
    """Returns the precomputed coloring for use with `jacrev`.

    This function performs the graph-coloring step that identifies structurally
    independent groups of output elements. Because it uses `scipy` and
    `networkx`, it is not JAX-traceable and must be called outside of
    `jax.jit`. The returned `ReverseModeColoring` can then be passed to
    `jacrev` via the `coloring` keyword argument, making `jacrev` itself
    callable inside a jitted function without needing
    `jax.ensure_compile_time_eval`.

    The `output_coloring` field of the returned object encodes a projection
    matrix `P` of shape `(ncolors, nrows)` where `P[c, i] = 1` iff
    `output_coloring[i] == c`. This matrix compresses the output before
    reverse-mode differentiation; the scatter-add
    `jnp.zeros(ncolors).at[output_coloring].add(y)` is equivalent to `P @ y`
    and avoids materialising the full 2-D matrix.

    Args:
        sparsity: Sparse matrix whose specified elements are at locations where
            the Jacobian is nonzero. Note that the values of `sparsity` are not
            used. Accepts either a JAX `BCOO` matrix (which must be rank-2 with
            `n_sparse == 2`) or a SciPy sparse matrix.
        coloring_strategy: See `networkx.algorithms.coloring.greedy_color`.

    Returns:
        A `ReverseModeColoring` containing the sparsity, output coloring
        vector, and number of colors.

    Raises:
        ValueError: If `sparsity` is not rank-2 or, for `BCOO` inputs,
            `sparsity.n_sparse != 2`.
    """
    if isinstance(sparsity, jsparse.BCOO):
        if sparsity.ndim != 2:
            raise ValueError(
                f"`sparsity` must be rank-2, but got shape of {sparsity.shape}."
            )
        if sparsity.n_sparse != 2:
            raise ValueError(
                f"`sparsity.n_sparse` must be 2, but got a value of "
                f"{sparsity.n_sparse}."
            )
        sparsity_bcoo = sparsity
        sparsity_scipy = ssparse.coo_matrix(
            (sparsity.data, sparsity.indices.T), shape=sparsity.shape
        )
    else:
        if sparsity.ndim != 2:
            raise ValueError(
                f"`sparsity` must be rank-2, but got shape of {sparsity.shape}."
            )
        sparsity_bcoo = jsparse.BCOO.from_scipy_sparse(sparsity)
        sparsity_scipy = sparsity

    connectivity = _output_connectivity_from_sparsity(sparsity_scipy)
    output_coloring_array, ncolors = _greedy_color(connectivity, coloring_strategy)
    assert output_coloring_array.size == sparsity_bcoo.shape[0]
    return ReverseModeColoring(
        sparsity=sparsity_bcoo,
        output_coloring=jnp.asarray(output_coloring_array),
        ncolors=StaticInt(ncolors),
    )


def jacfwd_coloring(
    sparsity: Union[jsparse.BCOO, ssparse.spmatrix],
    coloring_strategy: str = _DEFAULT_COLORING_STRATEGY,
) -> ForwardModeColoring:
    """Returns the precomputed coloring for use with `jacfwd`.

    This function performs the graph-coloring step that identifies structurally
    independent groups of input elements. Because it uses `scipy` and
    `networkx`, it is not JAX-traceable and must be called outside of
    `jax.jit`. The returned `ForwardModeColoring` can then be passed to
    `jacfwd` via the `coloring` keyword argument, making `jacfwd` itself
    callable inside a jitted function without needing
    `jax.ensure_compile_time_eval`.

    The `input_coloring` field of the returned object encodes a basis matrix
    `B` of shape `(ncols, ncolors)` where `B[j, c] = 1` iff
    `input_coloring[j] == c`. Column `c` of `B`, used as a JVP tangent vector,
    is equivalent to `(input_coloring == c).astype(float)`, so the vmap over
    colors can compute tangents on-the-fly and avoids materialising the full
    matrix.

    Args:
        sparsity: Sparse matrix whose specified elements are at locations where
            the Jacobian is nonzero. Note that the values of `sparsity` are not
            used. Accepts either a JAX `BCOO` matrix (which must be rank-2 with
            `n_sparse == 2`) or a SciPy sparse matrix.
        coloring_strategy: See `networkx.algorithms.coloring.greedy_color`.

    Returns:
        A `ForwardModeColoring` containing the sparsity, input coloring
        vector, and number of colors.

    Raises:
        ValueError: If `sparsity` is not rank-2 or, for `BCOO` inputs,
            `sparsity.n_sparse != 2`.
    """
    if isinstance(sparsity, jsparse.BCOO):
        if sparsity.ndim != 2:
            raise ValueError(
                f"`sparsity` must be rank-2, but got shape of {sparsity.shape}."
            )
        if sparsity.n_sparse != 2:
            raise ValueError(
                f"`sparsity.n_sparse` must be 2, but got a value of "
                f"{sparsity.n_sparse}."
            )
        sparsity_bcoo = sparsity
        sparsity_scipy = ssparse.coo_matrix(
            (sparsity.data, sparsity.indices.T), shape=sparsity.shape
        )
    else:
        if sparsity.ndim != 2:
            raise ValueError(
                f"`sparsity` must be rank-2, but got shape of {sparsity.shape}."
            )
        sparsity_bcoo = jsparse.BCOO.from_scipy_sparse(sparsity)
        sparsity_scipy = sparsity

    connectivity = _input_connectivity_from_sparsity(sparsity_scipy)
    input_coloring_array, ncolors = _greedy_color(connectivity, coloring_strategy)
    assert input_coloring_array.size == sparsity_bcoo.shape[1]
    return ForwardModeColoring(
        sparsity=sparsity_bcoo,
        input_coloring=jnp.asarray(input_coloring_array),
        ncolors=StaticInt(ncolors),
    )


def jacrev(
    fn: Callable[[Any], ArrayWithOptionalAux],
    sparsity: Union[jsparse.BCOO, None] = None,
    argnums: int = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    coloring_strategy: str = _DEFAULT_COLORING_STRATEGY,
    *,
    coloring: Union[ReverseModeColoring, None] = None,
) -> Callable[[Any], ArrayWithOptionalAux]:
    """Returns a function which computes the Jacobian of `fn` using reverse mode.

    This function uses reverse-mode automatic differentiation to compute the
    Jacobian. The `fn` must accept a rank-1 array and return a rank-1 array, and
    the Jacobian should be sparse with nonzero elements identified by `sparsity`.
    Sparsity is exploited in order to make the Jacobian computation efficient.

    This is done by identifying "structurally independent" groups of output
    elements, which is isomorphic to a graph coloring problem. This allows
    project to a lower-dimensional output space, so that reverse-mode
    differentiation can be more efficiently applied.

    Either `sparsity` or `coloring` must be provided, but not both. When
    `sparsity` is given, the coloring is computed internally via
    `jacrev_coloring`. When `coloring` is given (a precomputed
    `ReverseModeColoring`), the graph-coloring step is skipped, which allows
    this function to be used inside `jax.jit` without
    `jax.ensure_compile_time_eval`.

    Args:
        fn: The function for which the sparse Jacobian is sought. The function can
            have several inputs, one of which is to be differentiated with respect to.
            The argument top be differentiated with respect to should be a rank-1
            array with size equal to the column count in `sparsity`. The output of
            `fn` must be rank-1 with size equal to the row count in `sparsity`.
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero. Note that the values of `sparsity` are not used.
            Mutually exclusive with `coloring`.
        argnums: Specifies the positional argument to differentiate with respect to.
        has_aux: See `jax.jacrev`.
        holomorphic: See `jax.jacrev`.
        allow_int: See `jax.jacrev`.
        coloring_strategy: See `networkx.algorithms.coloring.greedy_color`. Only
            used when `sparsity` is provided.
        coloring: Precomputed coloring from `jacrev_coloring`. Mutually exclusive
            with `sparsity`.

    Returns:
        The function which computes the sparse Jacobian.

    Raises:
        ValueError: If both or neither of `sparsity` and `coloring` are provided,
            or if `argnums` is not an integer.
    """
    if sparsity is not None and coloring is not None:
        raise ValueError("Provide either `sparsity` or `coloring`, not both.")
    if sparsity is None and coloring is None:
        raise ValueError("Exactly one of `sparsity` or `coloring` must be provided.")
    if not isinstance(argnums, int):
        raise ValueError(f"`argnums` must be an integer, but got {argnums}.")
    if sparsity is not None:
        coloring = jacrev_coloring(sparsity, coloring_strategy)
    assert coloring is not None

    _sparsity = coloring.sparsity
    output_coloring = coloring.output_coloring
    ncolors = coloring.ncolors

    def jacrev_fn(*args: Any, **kwargs: Any) -> ArrayWithOptionalAux:
        x = args[argnums]
        if x.shape != (_sparsity.shape[1],):
            raise ValueError(
                f"`x` must be rank-1 with size matching the number of columns in "
                f"`sparsity`, but got shape {x.shape} when `sparsity` has shape "
                f"{_sparsity.shape}."
            )

        def _projected_fn(*args: Any, **kwargs: Any) -> ArrayWithOptionalAux:
            if has_aux:
                y, aux = fn(*args, **kwargs)
            else:
                y = fn(*args, **kwargs)
            if y.shape != (_sparsity.shape[0],):
                raise ValueError(
                    f"`fn(x)` must be rank-1 with size matching the number of rows in "
                    f"`sparsity`, but got shape {y.shape} when `sparsity` has shape "
                    f"{_sparsity.shape}."
                )
            # Equivalent to `P @ y` where P[c, i] = 1 iff output_coloring[i] == c.
            projected = jnp.zeros(ncolors, dtype=y.dtype).at[output_coloring].add(y)
            if has_aux:
                return projected, aux
            else:
                return projected

        compressed_jac_with_maybe_aux = jax.jacrev(
            _projected_fn,
            argnums=argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
        )(*args, **kwargs)

        if has_aux:
            compressed_jac, aux = compressed_jac_with_maybe_aux
            return _expand_jacrev_jac(compressed_jac, output_coloring, _sparsity), aux
        else:
            compressed_jac = compressed_jac_with_maybe_aux
            return _expand_jacrev_jac(compressed_jac, output_coloring, _sparsity)

    return jacrev_fn


def jacfwd(
    fn: Callable[[Any], ArrayWithOptionalAux],
    sparsity: Union[jsparse.BCOO, None] = None,
    argnums: int = 0,
    has_aux: bool = False,
    coloring_strategy: str = _DEFAULT_COLORING_STRATEGY,
    *,
    coloring: Union[ForwardModeColoring, None] = None,
) -> Callable[[Any], ArrayWithOptionalAux]:
    """Returns a function which computes the Jacobian of `fn` using forward mode.

    This function uses forward-mode automatic differentiation to compute the
    Jacobian. The `fn` must accept a rank-1 array and return a rank-1 array, and
    the Jacobian should be sparse with nonzero elements identified by `sparsity`.
    Sparsity is exploited in order to make the Jacobian computation efficient.

    This is done by identifying "structurally independent" groups of input
    elements, which is isomorphic to a graph coloring problem. This allows
    project to a lower-dimensional input space, so that forward-mode
    differentiation can be more efficiently applied.

    Either `sparsity` or `coloring` must be provided, but not both. When
    `sparsity` is given, the coloring is computed internally via
    `jacfwd_coloring`. When `coloring` is given (a precomputed
    `ForwardModeColoring`), the graph-coloring step is skipped, which allows
    this function to be used inside `jax.jit` without
    `jax.ensure_compile_time_eval`.

    Args:
        fn: The function for which the sparse Jacobian is sought. The function can
            have several inputs, one of which is to be differentiated with respect to.
            The argument top be differentiated with respect to should be a rank-1
            array with size equal to the column count in `sparsity`. The output of
            `fn` must be rank-1 with size equal to the row count in `sparsity`.
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero. Note that the values of `sparsity` are not used.
            Mutually exclusive with `coloring`.
        argnums: Specifies the positional argument to differentiate with respect to.
        has_aux: See `jax.jacfwd`.
        coloring_strategy: See `networkx.algorithms.coloring.greedy_color`. Only
            used when `sparsity` is provided.
        coloring: Precomputed coloring from `jacfwd_coloring`. Mutually exclusive
            with `sparsity`.

    Returns:
        The function which computes the sparse Jacobian.

    Raises:
        ValueError: If both or neither of `sparsity` and `coloring` are provided,
            or if `argnums` is not an integer.
    """
    if sparsity is not None and coloring is not None:
        raise ValueError("Provide either `sparsity` or `coloring`, not both.")
    if sparsity is None and coloring is None:
        raise ValueError("Exactly one of `sparsity` or `coloring` must be provided.")
    if not isinstance(argnums, int):
        raise ValueError(f"`argnums` must be an integer, but got {argnums}.")
    if sparsity is not None:
        coloring = jacfwd_coloring(sparsity, coloring_strategy)
    assert coloring is not None

    _sparsity = coloring.sparsity
    input_coloring = coloring.input_coloring
    ncolors = coloring.ncolors

    def jacfwd_fn(*args: Any, **kwargs: Any) -> ArrayWithOptionalAux:
        x = args[argnums]
        if x.shape != (_sparsity.shape[1],):
            raise ValueError(
                f"`x` must be rank-1 with size matching the number of columns in "
                f"`sparsity`, but got shape {x.shape} when `sparsity` has shape "
                f"{_sparsity.shape}."
            )

        def _fn(x: jnp.ndarray) -> ArrayWithOptionalAux:
            args_with_x = args[:argnums] + (x,) + args[argnums + 1 :]
            return fn(*args_with_x, **kwargs)

        if has_aux:
            # Equivalent to vmapping over columns of B where B[j, c] = 1 iff
            # input_coloring[j] == c.
            def _jvp_fn_with_aux(c: jnp.ndarray) -> Tuple[jnp.ndarray, Any]:
                tangents = (input_coloring == c).astype(float)
                _, tangents_out, aux = jax.jvp(_fn, (x,), (tangents,), has_aux=True)
                return tangents_out, aux

            compressed_jac_transpose, aux = jax.vmap(
                _jvp_fn_with_aux, out_axes=(0, None)
            )(jnp.arange(ncolors))

        else:
            # Equivalent to vmapping over columns of B where B[j, c] = 1 iff
            # input_coloring[j] == c.
            def _jvp_fn(c: jnp.ndarray) -> jnp.ndarray:
                tangents = (input_coloring == c).astype(float)
                _, tangents_out = jax.jvp(_fn, (x,), (tangents,), has_aux=False)
                return tangents_out

            compressed_jac_transpose = jax.vmap(_jvp_fn)(jnp.arange(ncolors))

        compressed_jac = compressed_jac_transpose.T
        if compressed_jac.shape != (_sparsity.shape[0], ncolors):
            raise ValueError(
                f"Got an invalid compressed Jacobian shape, which can occur if "
                f"`fn(x)` is not rank-1 with size matching the number of rows in "
                f"`sparsity`. Compressed Jacobian shape is {compressed_jac.shape} "
                f"when `sparsity` has shape {_sparsity.shape}."
            )

        if has_aux:
            return _expand_jacfwd_jac(compressed_jac, input_coloring, _sparsity), aux
        else:
            return _expand_jacfwd_jac(compressed_jac, input_coloring, _sparsity)

    return jacfwd_fn


# ------------------------------------------------------------------------------
# Private utility functions follow.
# ------------------------------------------------------------------------------


def _output_connectivity_from_sparsity(sparsity: ssparse.spmatrix) -> ssparse.spmatrix:
    """Computes the connectivity of output elements, given a Jacobian sparsity.

    Args:
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero.

    Returns:
        The sparse connectivity matrix for the output elements.
    """
    assert sparsity.ndim == 2
    return (sparsity @ sparsity.T).astype(bool)


def _input_connectivity_from_sparsity(sparsity: ssparse.spmatrix) -> ssparse.spmatrix:
    """Computes the connectivity of input elements, given a Jacobian sparsity.

    Args:
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero.

    Returns:
        The sparse connectivity matrix for the input elements.
    """
    assert sparsity.ndim == 2
    return (sparsity.T @ sparsity).astype(bool)


def _greedy_color(
    connectivity: ssparse.spmatrix,
    strategy: str,
) -> Tuple[onp.ndarray, int]:
    """Wraps `networkx.algorithms.coloring.greedy_color`.

    Args:
        connectivity: Sparse matrix giving the connectivity.
        strategy: The coloring strategy. See `networkx` documentation for details.

    Returns:
        A tuple containing the coloring vector and the number of colors used.
    """
    assert connectivity.ndim == 2
    assert connectivity.shape[0] == connectivity.shape[1]
    graph = networkx.convert_matrix.from_scipy_sparse_array(connectivity)
    coloring_dict = networkx.algorithms.coloring.greedy_color(graph, strategy)
    indices, colors = list(zip(*coloring_dict.items()))
    coloring = onp.asarray(colors)[onp.argsort(indices)]
    return coloring, onp.unique(coloring).size


def _expand_jacrev_jac(
    compressed_jac: jnp.ndarray,
    output_coloring: jnp.ndarray,
    sparsity: jsparse.BCOO,
) -> jsparse.BCOO:
    """Expands an output-compressed Jacobian into a sparse matrix.

    Args:
        compressed_jac: The compressed Jacobian.
        output_coloring: Coloring of the output elements.
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero.

    Returns:
        The sparse Jacobian matrix.
    """
    assert compressed_jac.ndim == 2
    assert output_coloring.ndim == 1
    assert sparsity.shape == (output_coloring.size, compressed_jac.shape[1])
    row, col = sparsity.indices.T
    compressed_index = (output_coloring[row], col)
    data = compressed_jac[compressed_index]
    return jsparse.BCOO((data, sparsity.indices), shape=sparsity.shape)


def _expand_jacfwd_jac(
    compressed_jac: jnp.ndarray,
    input_coloring: jnp.ndarray,
    sparsity: jsparse.BCOO,
) -> jsparse.BCOO:
    """Expands an input-compressed Jacobian into a sparse matrix.

    Args:
        compressed_jac: The compressed Jacobian.
        input_coloring: Coloring of the input elements.
        sparsity: Sparse matrix whose specified elements are at locations where the
            Jacobian is nonzero.

    Returns:
        The sparse Jacobian matrix.
    """
    assert compressed_jac.ndim == 2
    assert input_coloring.ndim == 1
    assert sparsity.shape == (compressed_jac.shape[0], input_coloring.size)
    row, col = sparsity.indices.T
    compressed_index = (row, input_coloring[col])
    data = compressed_jac[compressed_index]
    return jsparse.BCOO((data, sparsity.indices), shape=sparsity.shape)
