"""Defines functions for efficient computation of sparse Jacobians."""

from typing import Callable, Tuple, Union

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import networkx
import numpy as onp
import scipy.sparse as ssparse


# Coloring strategy employed to find structurally-independent output elements.
_DEFAULT_COLORING_STRATEGY = 'largest_first'


def jacrev(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    sparsity: jax.experimental.sparse.BCOO,
    coloring_strategy: str = _DEFAULT_COLORING_STRATEGY,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns a function which computes the Jacobian of `fn` using reverse mode.

  This function uses reverse-mode automatic differentiation to compute the
  Jacobian. The `fn` must accept a rank-1 array and return a rank-1 array, and
  the Jacobian should be sparse with nonzero elements identified by `sparsity`.
  Sparsity is exploited in order to make the Jacobian computation efficient.

  This is done by identifying "structurally independent" groups of output
  elements, which is isomorphic to a graph coloring problem. This allows
  project to a lower-dimensional output space, so that reverse-mode
  differentiation can be more efficiently applied.

  Args:
    fn: The function for which the sparse Jacobian is sought. Inputs to `fn`
      should be rank-1 with size equal to the number of columns in `sparsity`.
      Outputs should be rank-1 with size equal to the number of rows.
    sparsity: Sparse matrix whose specified elements are at locations where the
      Jacobian is nonzero. Note that the values of `sparsity` are not used.
    coloring_strategy: See `networkx.algorithms.coloring.greedy_color`.

  Returns:
    The function which computes the sparse Jacobian.
  """
  if sparsity.ndim != 2:
    raise ValueError(
        f'`sparsity` must be rank-2, but got shape of {sparsity.shape}')
  
  # Identify the structurally-independent elements of `fn` output, i.e. obtain
  # the coloring of the output. Here we must use `scipy` sparse matrices.
  sparsity_scipy = ssparse.coo_matrix(
      (sparsity.data, sparsity.indices.T), shape=sparsity.shape)
  connectivity = _output_connectivity_from_sparsity(sparsity_scipy)
  output_coloring, ncolors = _greedy_color(connectivity, coloring_strategy)
  output_coloring = jnp.asarray(output_coloring)
  assert output_coloring.size == sparsity.shape[0]

  projection_matrix = (
      jnp.arange(ncolors)[:, jnp.newaxis] == output_coloring[jnp.newaxis, :])
  projection_matrix = projection_matrix.astype(jnp.float32)
  
  def jacrev_fn(x):
    if x.shape != (sparsity.shape[1],):
      raise ValueError(
          f'`x` must be rank-1 with size matching the number of columns in '
          f'`sparsity`, but got shape {x.shape} when `sparsity` has shape '
          f'{sparsity.shape}.')
    
    def _projected_fn(x):
      y = fn(x)
      if y.shape != (sparsity.shape[0],):
        raise ValueError(
            f'`fn(x)` must be rank-1 with size matching the number of rows in '
            f'`sparsity`, but got shape {y.shape} when `sparsity` has shape '
            f'{sparsity.shape}.')
      return projection_matrix @ y

    compressed_jac = jax.jacrev(_projected_fn)(x)
    return _expand_jacrev_jac(compressed_jac, output_coloring, sparsity)

  return jacrev_fn


def jacfwd(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    sparsity: jax.experimental.sparse.BCOO,
    coloring_strategy: str = _DEFAULT_COLORING_STRATEGY,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns a function which computes the Jacobian of `fn` using forward mode.

  This function uses forward-mode automatic differentiation to compute the
  Jacobian. The `fn` must accept a rank-1 array and return a rank-1 array, and
  the Jacobian should be sparse with nonzero elements identified by `sparsity`.
  Sparsity is exploited in order to make the Jacobian computation efficient.

  This is done by identifying "structurally independent" groups of input
  elements, which is isomorphic to a graph coloring problem. This allows
  project to a lower-dimensional input space, so that forward-mode
  differentiation can be more efficiently applied.

  Args:
    fn: The function for which the sparse Jacobian is sought. Inputs to `fn`
      should be rank-1 with size equal to the number of columns in `sparsity`.
      Outputs should be rank-1 with size equal to the number of rows.
    sparsity: Sparse matrix whose specified elements are at locations where the
      Jacobian is nonzero. Note that the values of `sparsity` are not used.
    coloring_strategy: See `networkx.algorithms.coloring.greedy_color`.

  Returns:
    The function which computes the sparse Jacobian.
  """
  if sparsity.ndim != 2:
    raise ValueError(
        f'`sparsity` must be rank-2, but got shape of {sparsity.shape}')
  
  # Identify the structurally-independent elements of `fn` output, i.e. obtain
  # the coloring of the output. Here we must use `scipy` sparse matrices.
  sparsity_scipy = ssparse.coo_matrix(
      (sparsity.data, sparsity.indices.T), shape=sparsity.shape)
  connectivity = _input_connectivity_from_sparsity(sparsity_scipy)
  input_coloring, ncolors = _greedy_color(connectivity, coloring_strategy)
  input_coloring = jnp.asarray(input_coloring)
  assert input_coloring.size == sparsity.shape[1]

  projection_matrix = (
      jnp.arange(ncolors)[jnp.newaxis, :] == input_coloring[:, jnp.newaxis])
  projection_matrix = projection_matrix.astype(jnp.float32)
  
  basis = (
      jnp.arange(ncolors)[jnp.newaxis, :] == input_coloring[:, jnp.newaxis])
  basis = basis.astype(jnp.float32)
  
  def jacfwd_fn(x):
    if x.shape != (sparsity.shape[1],):
      raise ValueError(
          f'`x` must be rank-1 with size matching the number of columns in '
          f'`sparsity`, but got shape {x.shape} when `sparsity` has shape '
          f'{sparsity.shape}.')

    _jvp = lambda s: jax.jvp(fn, (x,), (s,))[1]
    compressed_jac_transpose = jax.vmap(_jvp, in_axes=1)(basis)
    compressed_jac = compressed_jac_transpose.T

    if compressed_jac.shape != (sparsity.shape[0], ncolors):
      raise ValueError(
          f'Got an invalid compressed Jacobian shape, which can occur if '
          f'`fn(x)` is not rank-1 with size matching the number of rows in '
          f'`sparsity`. Compressed Jacobian shape is {compressed_jac.shape} '
          f'when `sparsity` has shape {sparsity.shape}.')

    return _expand_jacfwd_jac(compressed_jac, input_coloring, sparsity)

  return jacfwd_fn


def _output_connectivity_from_sparsity(
    sparsity: ssparse.spmatrix) -> ssparse.spmatrix:
  """Computes the connectivity of output elements, given a Jacobian sparsity.
  
  Args:
    sparsity: Sparse matrix whose specified elements are at locations where the
      Jacobian is nonzero.
  
  Returns:
    The sparse connectivity matrix for the output elements.
  """
  assert sparsity.ndim == 2
  return (sparsity @ sparsity.T).astype(bool)


def _input_connectivity_from_sparsity(
    sparsity: ssparse.spmatrix) -> ssparse.spmatrix:
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
  graph = networkx.convert_matrix.from_scipy_sparse_matrix(connectivity)
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
