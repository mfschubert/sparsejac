"""Defines a function for efficient computation of sparse Jacobians."""

from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import networkx
import numpy as onp
import scipy.sparse as ssparse


# Coloring strategy employed to find strucuturally-independent output elements.
_DEFAULT_COLORING_STRATEGY = 'largest_first'


def jacrev_sparse(
    fn: Callable[[jnp.ndarray], jnp.ndarray],
    sparsity: jax.experimental.sparse.BCOO,
    coloring_strategy: str = _DEFAULT_COLORING_STRATEGY,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns a function which computes the sparse Jacobian of `fn`.

  The `fn` must accept a rank-1 array and return a rank-1 array, and the
  Jacobian should be sparse with nonzero elements identified by `sparsity`. The
  sparsity is exploited in order to make the Jacobian computation efficient.

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
  connectivity = _connectivity_from_sparsity(sparsity_scipy)
  coloring, ncolors = _greedy_color(connectivity, coloring_strategy)
  coloring = jnp.asarray(coloring)
  assert coloring.size == sparsity.shape[0]

  projection_matrix = (
      jnp.arange(ncolors)[:, jnp.newaxis] == coloring[jnp.newaxis, :])
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
    return _expand_jac(compressed_jac, coloring, sparsity)

  return jacrev_fn


def _connectivity_from_sparsity(sparsity: ssparse.spmatrix) -> ssparse.spmatrix:
  """Computes the connectivity of output elements, given a Jacobian sparsity.
  
  Args:
    sparsity: Sparse matrix whose specified elements are at locations where the
      Jacobian is nonzero.
  
  Returns:
    The sparse connectivity matrix for the output elements.
  """
  assert sparsity.ndim == 2
  return (sparsity @ sparsity.T).astype(bool)


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


def _expand_jac(
    compressed_jac: jnp.ndarray,
    coloring: jnp.ndarray,
    sparsity: jsparse.BCOO,
) -> jsparse.BCOO:
  """Expands a compressed Jacobian into a sparse matrix.
  
  Args:
    compressed_jac: The compressed Jacobian.
    coloring: Coloring of the residue elements.
    sparsity: Sparse matrix whose specified elements are at locations where the
      Jacobian is nonzero.

  Returns:
    The sparse Jacobian matrix.
  """
  assert compressed_jac.ndim == 2
  assert coloring.ndim == 1
  assert sparsity.shape == (coloring.size, compressed_jac.shape[1])
  row, col = sparsity.indices.T
  compressed_index = (coloring[row], col)
  data = compressed_jac[compressed_index]
  return jsparse.BCOO((data, sparsity.indices), shape=sparsity.shape)