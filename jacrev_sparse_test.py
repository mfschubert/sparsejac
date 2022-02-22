"""Tests for `jacrev_sparse`."""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
import networkx
import numpy as onp
import scipy.sparse as ssparse
import unittest

import jacrev_sparse


_SIZE = 50


class JacrevSparseTest(unittest.TestCase):

  def test_sparsity_shape_validation(self):
    with self.assertRaisesRegex(
        ValueError, '`sparsity` must be rank-2, but got shape'):
      invalid_sparsity = jsparse.BCOO.fromdense(jnp.ones((5, 5, 5)))
      jacrev_sparse.jacrev_sparse(lambda x: x, invalid_sparsity)

  def test_input_shape_validation(self):
    sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
    jacfn = jacrev_sparse.jacrev_sparse(lambda x: x, sparsity)
    with self.assertRaisesRegex(
        ValueError, '`x` must be rank-1 with size matching'):
      jacfn(jnp.ones((10, 5)))

  def test_output_shape_validation(self):
    sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
    invalid_fn = lambda x: jnp.reshape(x, (10, 5))
    jacfn = jacrev_sparse.jacrev_sparse(invalid_fn, sparsity)
    with self.assertRaisesRegex(
        ValueError, '`fn\(x\)` must be rank-1 with size matching'):
      jacfn(jnp.ones(_SIZE))

  def test_diagonal(self):
    fn = lambda x: x**2
    sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    actual = jacrev_sparse.jacrev_sparse(fn, sparsity)(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())
  
  def test_diagonal_jit(self):
    fn = lambda x: x**2
    sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    jacfn = jacrev_sparse.jacrev_sparse(fn, sparsity)
    jacfn = jax.jit(jacfn)
    actual = jacfn(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

  def test_diagonal_shuffled(self):
    fn = lambda x: jax.random.permutation(jax.random.PRNGKey(0), x**2)
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    expected = jax.jacrev(fn)(x)
    sparsity = jsparse.BCOO.fromdense(expected != 0)
    actual = jacrev_sparse.jacrev_sparse(fn, sparsity)(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

  def test_dense(self):
    fn = lambda x: jnp.stack((jnp.sum(x), jnp.sum(x)**2, jnp.sum(x)**3))
    sparsity = jsparse.BCOO.fromdense(jnp.ones((3, _SIZE)))
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    actual = jacrev_sparse.jacrev_sparse(fn, sparsity)(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

  def test_convolutional_1d(self):
    fn = lambda x: jnp.convolve(x, jnp.asarray([1., -2., 1.]), mode='valid')
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
    i, j = jnp.meshgrid(jnp.arange(_SIZE - 2), jnp.arange(_SIZE), indexing='ij')
    sparsity = (i == j) | ((i + 1) == j) | ((i + 2) == j)
    sparsity = jsparse.BCOO.fromdense(sparsity)
    actual = jacrev_sparse.jacrev_sparse(fn, sparsity)(x)
    onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

  def test_convolutional_2d(self):
    shape_2d = (20, 20)

    def fn(x_flat):
      x = jnp.reshape(x_flat, shape_2d)
      result = jax.scipy.signal.convolve2d(x, jnp.ones((3, 3)), mode='valid')
      return result.flatten()

    x_flat = jax.random.uniform(
        jax.random.PRNGKey(0), shape=(shape_2d[0] * shape_2d[1],))
    expected = jax.jacrev(fn)(x_flat)
    sparsity = jsparse.BCOO.fromdense(expected != 0)
    actual = jacrev_sparse.jacrev_sparse(fn, sparsity)(x_flat)
    onp.testing.assert_array_equal(expected, actual.todense())


class ConnectivityFromSparsityTest(unittest.TestCase):
  
  def test_connectivity_matches_expected(self):
    sparsity = onp.asarray(
        [[1, 1, 1, 0, 0, 0],
         [0, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1],
         [1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1]])
    sparsity = ssparse.coo_matrix(sparsity)
    expected = jnp.asarray(
        [[1, 1, 1, 0, 1, 1],
         [1, 1, 1, 1, 0, 1],
         [1, 1, 1, 1, 0, 1],
         [0, 1, 1, 1, 0, 1],
         [1, 0, 0, 0, 1, 1],
         [1, 1, 1, 1, 1, 1]])
    actual = jacrev_sparse._connectivity_from_sparsity(sparsity)
    onp.testing.assert_array_equal(expected, actual.todense())


if __name__ == '__main__':
  unittest.main(argv=[''], verbosity=2, exit=False)