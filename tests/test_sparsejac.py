"""Tests for `sparsejac`."""

import unittest

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as onp
import scipy.sparse as ssparse

from sparsejac import sparsejac

_SIZE = 50


class JacrevTest(unittest.TestCase):
    def test_sparsity_shape_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`sparsity` must be rank-2, but got shape"
        ):
            invalid_sparsity = jsparse.BCOO.fromdense(jnp.ones((5, 5, 5)))
            sparsejac.jacrev(lambda x: x, invalid_sparsity)

    def test_sparsity_n_sparse_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`sparsity.n_sparse` must be 2, but got a value of"
        ):
            data = jnp.ones((5, 5))
            indices = jnp.arange(5)[:, jnp.newaxis]
            invalid_sparsity = jsparse.BCOO((data, indices), shape=(5, 5))
            assert invalid_sparsity.ndim == 2
            assert invalid_sparsity.n_sparse == 1
            sparsejac.jacrev(lambda x: x, invalid_sparsity)

    def test_input_shape_validation(self):
        sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
        jacfn = sparsejac.jacrev(lambda x: x, sparsity)
        with self.assertRaisesRegex(
            ValueError, "`x` must be rank-1 with size matching"
        ):
            jacfn(jnp.ones((10, 5)))

    def test_output_shape_validation(self):
        sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
        invalid_fn = lambda x: jnp.reshape(x, (10, 5))
        jacfn = sparsejac.jacrev(invalid_fn, sparsity)
        with self.assertRaisesRegex(
            ValueError, r"`fn\(x\)` must be rank-1 with size matching"
        ):
            jacfn(jnp.ones(_SIZE))

    def test_argnums_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`argnums` must be an integer, but got"
        ):
            sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
            sparsejac.jacrev(lambda x: x, sparsity, argnums=(0, 1))

    def test_diagonal(self):
        fn = lambda x: x**2
        sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        actual = sparsejac.jacrev(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_diagonal_jit(self):
        fn = lambda x: x**2
        sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        jacfn = sparsejac.jacrev(fn, sparsity)
        jacfn = jax.jit(jacfn)
        actual = jacfn(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_diagonal_shuffled(self):
        fn = lambda x: jax.random.permutation(jax.random.PRNGKey(0), x**2)
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        expected = jax.jacrev(fn)(x)
        sparsity = jsparse.BCOO.fromdense(expected != 0)
        actual = sparsejac.jacrev(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_dense(self):
        fn = lambda x: jnp.stack((jnp.sum(x), jnp.sum(x) ** 2, jnp.sum(x) ** 3))
        sparsity = jsparse.BCOO.fromdense(jnp.ones((3, _SIZE)))
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        actual = sparsejac.jacrev(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_convolutional_1d(self):
        fn = lambda x: jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="valid")
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        i, j = jnp.meshgrid(jnp.arange(_SIZE - 2), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i + 1) == j) | ((i + 2) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)
        actual = sparsejac.jacrev(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_convolutional_1d_nonlinear(self):
        fn = lambda x: jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="valid") ** 2
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        i, j = jnp.meshgrid(jnp.arange(_SIZE - 2), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i + 1) == j) | ((i + 2) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)
        actual = sparsejac.jacrev(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_convolutional_2d(self):
        shape_2d = (20, 20)

        def fn(x_flat):
            x = jnp.reshape(x_flat, shape_2d)
            result = jax.scipy.signal.convolve2d(x, jnp.ones((3, 3)), mode="valid")
            return result.flatten()

        x_flat = jax.random.uniform(
            jax.random.PRNGKey(0), shape=(shape_2d[0] * shape_2d[1],)
        )
        expected = jax.jacrev(fn)(x_flat)
        sparsity = jsparse.BCOO.fromdense(expected != 0)
        actual = sparsejac.jacrev(fn, sparsity)(x_flat)
        onp.testing.assert_array_equal(expected, actual.todense())

    def test_convolutional_2d_nonlinear(self):
        shape_2d = (20, 20)

        def fn(x_flat):
            x = jnp.reshape(x_flat, shape_2d)
            result = jax.scipy.signal.convolve2d(x, jnp.ones((3, 3)), mode="valid")
            return result.flatten() ** 2

        x_flat = jax.random.uniform(
            jax.random.PRNGKey(0), shape=(shape_2d[0] * shape_2d[1],)
        )
        expected = jax.jacrev(fn)(x_flat)
        sparsity = jsparse.BCOO.fromdense(expected != 0)
        actual = sparsejac.jacrev(fn, sparsity)(x_flat)
        onp.testing.assert_array_equal(expected, actual.todense())

    def test_argnums(self):
        def fn(x, y, z):
            convolved = jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="same") ** 2
            return y * convolved + z

        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        y = jax.random.uniform(jax.random.PRNGKey(1), shape=(_SIZE,))
        z = jax.random.uniform(jax.random.PRNGKey(2), shape=(_SIZE,))

        i, j = jnp.meshgrid(jnp.arange(_SIZE), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i - 1) == j) | ((i + 1) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)

        with self.subTest():
            result = sparsejac.jacrev(fn, sparsity, argnums=0)(x, y, z)
            expected = jax.jacrev(fn, argnums=0)(x, y, z)
            onp.testing.assert_array_equal(expected, result.todense())

        with self.subTest():
            result = sparsejac.jacrev(fn, sparsity, argnums=1)(x, y, z)
            expected = jax.jacrev(fn, argnums=1)(x, y, z)
            onp.testing.assert_array_equal(expected, result.todense())

        with self.subTest():
            result = sparsejac.jacrev(fn, sparsity, argnums=2)(x, y, z)
            expected = jax.jacrev(fn, argnums=2)(x, y, z)
            onp.testing.assert_array_equal(expected, result.todense())

    def test_has_aux(self):
        def fn(x):
            convolved = jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="same") ** 2
            aux = x + 1
            return convolved, aux

        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        i, j = jnp.meshgrid(jnp.arange(_SIZE), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i - 1) == j) | ((i + 1) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)

        result_jac, result_aux = sparsejac.jacrev(fn, sparsity, has_aux=True)(x)
        expected_jac, expected_aux = jax.jacrev(fn, has_aux=True)(x)
        onp.testing.assert_array_equal(expected_jac, result_jac.todense())
        onp.testing.assert_array_equal(expected_aux, result_aux)

    def test_kwargs(self):
        def fn(x, y):
            return y * jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="same") ** 2

        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        i, j = jnp.meshgrid(jnp.arange(_SIZE), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i - 1) == j) | ((i + 1) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)

        result_jac = sparsejac.jacrev(fn, sparsity)(x, y=1)
        expected_jac = jax.jacrev(fn)(x, y=1)
        onp.testing.assert_array_equal(expected_jac, result_jac.todense())


class JacfwdTest(unittest.TestCase):
    def test_sparsity_shape_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`sparsity` must be rank-2, but got shape"
        ):
            invalid_sparsity = jsparse.BCOO.fromdense(jnp.ones((5, 5, 5)))
            sparsejac.jacfwd(lambda x: x, invalid_sparsity)

    def test_sparsity_n_sparse_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`sparsity.n_sparse` must be 2, but got a value of"
        ):
            data = jnp.ones((5, 5))
            indices = jnp.arange(5)[:, jnp.newaxis]
            invalid_sparsity = jsparse.BCOO((data, indices), shape=(5, 5))
            assert invalid_sparsity.ndim == 2
            assert invalid_sparsity.n_sparse == 1
            sparsejac.jacfwd(lambda x: x, invalid_sparsity)

    def test_input_shape_validation(self):
        sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
        jacfn = sparsejac.jacfwd(lambda x: x, sparsity)
        with self.assertRaisesRegex(
            ValueError, "`x` must be rank-1 with size matching"
        ):
            jacfn(jnp.ones((10, 5)))

    def test_output_shape_validation(self):
        sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
        invalid_fn = lambda x: jnp.reshape(x, (10, 5))
        jacfn = sparsejac.jacfwd(invalid_fn, sparsity)
        with self.assertRaisesRegex(
            ValueError, "Got an invalid compressed Jacobian shape, which can "
        ):
            jacfn(jnp.ones(_SIZE))

    def test_argnums_validation(self):
        with self.assertRaisesRegex(
            ValueError, "`argnums` must be an integer, but got"
        ):
            sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
            sparsejac.jacfwd(lambda x: x, sparsity, argnums=(0, 1))

    def test_diagonal(self):
        fn = lambda x: x**2
        sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        actual = sparsejac.jacfwd(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_diagonal_jit(self):
        fn = lambda x: x**2
        sparsity = jsparse.BCOO.fromdense(jnp.eye(_SIZE))
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        jacfn = sparsejac.jacfwd(fn, sparsity)
        jacfn = jax.jit(jacfn)
        actual = jacfn(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_diagonal_shuffled(self):
        fn = lambda x: jax.random.permutation(jax.random.PRNGKey(0), x**2)
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        expected = jax.jacrev(fn)(x)
        sparsity = jsparse.BCOO.fromdense(expected != 0)
        actual = sparsejac.jacfwd(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_dense(self):
        fn = lambda x: jnp.stack((jnp.sum(x), jnp.sum(x) ** 2, jnp.sum(x) ** 3))
        sparsity = jsparse.BCOO.fromdense(jnp.ones((3, _SIZE)))
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        actual = sparsejac.jacfwd(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_convolutional_1d(self):
        fn = lambda x: jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="valid")
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        i, j = jnp.meshgrid(jnp.arange(_SIZE - 2), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i + 1) == j) | ((i + 2) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)
        actual = sparsejac.jacfwd(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_convolutional_1d_nonlinear(self):
        fn = lambda x: jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="valid") ** 2
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        i, j = jnp.meshgrid(jnp.arange(_SIZE - 2), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i + 1) == j) | ((i + 2) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)
        actual = sparsejac.jacfwd(fn, sparsity)(x)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), actual.todense())

    def test_convolutional_2d(self):
        shape_2d = (20, 20)

        def fn(x_flat):
            x = jnp.reshape(x_flat, shape_2d)
            result = jax.scipy.signal.convolve2d(x, jnp.ones((3, 3)), mode="valid")
            return result.flatten()

        x_flat = jax.random.uniform(
            jax.random.PRNGKey(0), shape=(shape_2d[0] * shape_2d[1],)
        )
        expected = jax.jacrev(fn)(x_flat)
        sparsity = jsparse.BCOO.fromdense(expected != 0)
        actual = sparsejac.jacfwd(fn, sparsity)(x_flat)
        onp.testing.assert_array_equal(expected, actual.todense())

    def test_convolutional_2d_nonlinear(self):
        shape_2d = (20, 20)

        def fn(x_flat):
            x = jnp.reshape(x_flat, shape_2d)
            result = jax.scipy.signal.convolve2d(x, jnp.ones((3, 3)), mode="valid")
            return result.flatten() ** 2

        x_flat = jax.random.uniform(
            jax.random.PRNGKey(0), shape=(shape_2d[0] * shape_2d[1],)
        )
        expected = jax.jacrev(fn)(x_flat)
        sparsity = jsparse.BCOO.fromdense(expected != 0)
        actual = sparsejac.jacfwd(fn, sparsity)(x_flat)
        onp.testing.assert_array_equal(expected, actual.todense())

    def test_argnums(self):
        def fn(x, y, z):
            convolved = jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="same") ** 2
            return y * convolved + z

        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        y = jax.random.uniform(jax.random.PRNGKey(1), shape=(_SIZE,))
        z = jax.random.uniform(jax.random.PRNGKey(2), shape=(_SIZE,))

        i, j = jnp.meshgrid(jnp.arange(_SIZE), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i - 1) == j) | ((i + 1) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)

        with self.subTest():
            result = sparsejac.jacfwd(fn, sparsity, argnums=0)(x, y, z)
            expected = jax.jacfwd(fn, argnums=0)(x, y, z)
            onp.testing.assert_array_equal(expected, result.todense())

        with self.subTest():
            result = sparsejac.jacfwd(fn, sparsity, argnums=1)(x, y, z)
            expected = jax.jacfwd(fn, argnums=1)(x, y, z)
            onp.testing.assert_array_equal(expected, result.todense())

        with self.subTest():
            result = sparsejac.jacfwd(fn, sparsity, argnums=2)(x, y, z)
            expected = jax.jacfwd(fn, argnums=2)(x, y, z)
            onp.testing.assert_array_equal(expected, result.todense())

    def test_has_aux(self):
        def fn(x):
            convolved = jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="same") ** 2
            aux = x + 1
            return convolved, aux

        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        i, j = jnp.meshgrid(jnp.arange(_SIZE), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i - 1) == j) | ((i + 1) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)

        result_jac, result_aux = sparsejac.jacfwd(fn, sparsity, has_aux=True)(x)
        expected_jac, expected_aux = jax.jacfwd(fn, has_aux=True)(x)
        onp.testing.assert_array_equal(expected_jac, result_jac.todense())
        onp.testing.assert_array_equal(expected_aux, result_aux)

    def test_kwargs(self):
        def fn(x, y):
            return y * jnp.convolve(x, jnp.asarray([1.0, -2.0, 1.0]), mode="same") ** 2

        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        i, j = jnp.meshgrid(jnp.arange(_SIZE), jnp.arange(_SIZE), indexing="ij")
        sparsity = (i == j) | ((i - 1) == j) | ((i + 1) == j)
        sparsity = jsparse.BCOO.fromdense(sparsity)

        result_jac = sparsejac.jacfwd(fn, sparsity)(x, y=1)
        expected_jac = jax.jacfwd(fn)(x, y=1)
        onp.testing.assert_array_equal(expected_jac, result_jac.todense())


class ConnectivityFromSparsityTest(unittest.TestCase):
    def test_output_connectivity_matches_expected(self):
        sparsity = onp.asarray(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        )
        sparsity = ssparse.coo_matrix(sparsity)
        expected = jnp.asarray(
            [
                [1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0, 1],
                [0, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        )
        actual = sparsejac._output_connectivity_from_sparsity(sparsity)
        onp.testing.assert_array_equal(expected, actual.todense())

    def test_input_connectivity_matches_expected(self):
        sparsity = onp.asarray(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        sparsity = ssparse.coo_matrix(sparsity)
        expected = jnp.asarray(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        actual = sparsejac._input_connectivity_from_sparsity(sparsity)
        onp.testing.assert_array_equal(expected, actual.todense())


class JacrevColoringTest(unittest.TestCase):
    def _diagonal_sparsity(self) -> jsparse.BCOO:
        return jsparse.BCOO.fromdense(jnp.eye(_SIZE))

    def test_jacrev_coloring_returns_correct_type(self):
        sparsity = self._diagonal_sparsity()
        coloring = sparsejac.jacrev_coloring(sparsity)
        self.assertIsInstance(coloring, sparsejac.ReverseModeColoring)

    def test_jacrev_coloring_fields(self):
        sparsity = self._diagonal_sparsity()
        coloring = sparsejac.jacrev_coloring(sparsity)
        self.assertEqual(coloring.output_coloring.shape, (sparsity.shape[0],))
        self.assertGreater(coloring.ncolors, 0)
        self.assertLessEqual(coloring.ncolors, sparsity.shape[0])
        onp.testing.assert_array_equal(
            jnp.unique(coloring.output_coloring).size, coloring.ncolors
        )

    def test_jacrev_with_precomputed_coloring_matches_sparsity_api(self):
        fn = lambda x: x**2
        sparsity = self._diagonal_sparsity()
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        expected = sparsejac.jacrev(fn, sparsity)(x)
        coloring = sparsejac.jacrev_coloring(sparsity)
        actual = sparsejac.jacrev(fn, coloring=coloring)(x)
        onp.testing.assert_array_equal(expected.todense(), actual.todense())

    def test_jacrev_with_precomputed_coloring_jit(self):
        fn = lambda x: x**2
        sparsity = self._diagonal_sparsity()
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        coloring = sparsejac.jacrev_coloring(sparsity)

        @jax.jit
        def jacrev_sparse(x, coloring):
            jacfn = sparsejac.jacrev(fn, coloring=coloring)
            return jacfn(x)
        
        jac = jacrev_sparse(x, coloring)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), jac.todense())

    def test_jacrev_neither_sparsity_nor_coloring_raises(self):
        with self.assertRaisesRegex(ValueError, "Exactly one of"):
            sparsejac.jacrev(lambda x: x)

    def test_jacrev_both_sparsity_and_coloring_raises(self):
        sparsity = self._diagonal_sparsity()
        coloring = sparsejac.jacrev_coloring(sparsity)
        with self.assertRaisesRegex(ValueError, "not both"):
            sparsejac.jacrev(lambda x: x, sparsity, coloring=coloring)

    def test_jacrev_coloring_sparsity_shape_validation(self):
        with self.assertRaisesRegex(ValueError, "`sparsity` must be rank-2"):
            invalid_sparsity = jsparse.BCOO.fromdense(jnp.ones((5, 5, 5)))
            sparsejac.jacrev_coloring(invalid_sparsity)

    def test_jacrev_coloring_accepts_scipy_sparse(self):
        import scipy.sparse as ssparse

        sparsity_scipy = ssparse.eye(_SIZE, format="coo")
        coloring = sparsejac.jacrev_coloring(sparsity_scipy)
        self.assertIsInstance(coloring, sparsejac.ReverseModeColoring)
        self.assertEqual(coloring.output_coloring.shape, (_SIZE,))

    def test_jacrev_with_scipy_coloring_matches_jax_coloring(self):
        import scipy.sparse as ssparse

        fn = lambda x: x**2
        sparsity_bcoo = self._diagonal_sparsity()
        sparsity_scipy = ssparse.eye(_SIZE, format="coo")
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        result_bcoo = sparsejac.jacrev(fn, coloring=sparsejac.jacrev_coloring(sparsity_bcoo))(x)
        result_scipy = sparsejac.jacrev(fn, coloring=sparsejac.jacrev_coloring(sparsity_scipy))(x)
        onp.testing.assert_array_equal(result_bcoo.todense(), result_scipy.todense())


class JacfwdColoringTest(unittest.TestCase):
    def _diagonal_sparsity(self) -> jsparse.BCOO:
        return jsparse.BCOO.fromdense(jnp.eye(_SIZE))

    def test_jacfwd_coloring_returns_correct_type(self):
        sparsity = self._diagonal_sparsity()
        coloring = sparsejac.jacfwd_coloring(sparsity)
        self.assertIsInstance(coloring, sparsejac.ForwardModeColoring)

    def test_jacfwd_coloring_fields(self):
        sparsity = self._diagonal_sparsity()
        coloring = sparsejac.jacfwd_coloring(sparsity)
        self.assertEqual(coloring.input_coloring.shape, (sparsity.shape[1],))
        self.assertGreater(coloring.ncolors, 0)
        self.assertLessEqual(coloring.ncolors, sparsity.shape[1])
        onp.testing.assert_array_equal(
            jnp.unique(coloring.input_coloring).size, coloring.ncolors
        )

    def test_jacfwd_with_precomputed_coloring_matches_sparsity_api(self):
        fn = lambda x: x**2
        sparsity = self._diagonal_sparsity()
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        expected = sparsejac.jacfwd(fn, sparsity)(x)
        coloring = sparsejac.jacfwd_coloring(sparsity)
        actual = sparsejac.jacfwd(fn, coloring=coloring)(x)
        onp.testing.assert_array_equal(expected.todense(), actual.todense())

    def test_jacfwd_with_precomputed_coloring_jit(self):
        fn = lambda x: x**2
        sparsity = self._diagonal_sparsity()
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        coloring = sparsejac.jacfwd_coloring(sparsity)

        @jax.jit
        def jacfwd_sparse(x, coloring):
            jacfn = sparsejac.jacfwd(fn, coloring=coloring)
            return jacfn(x)
        
        jac = jacfwd_sparse(x, coloring)
        onp.testing.assert_array_equal(jax.jacrev(fn)(x), jac.todense())

    def test_jacfwd_neither_sparsity_nor_coloring_raises(self):
        with self.assertRaisesRegex(ValueError, "Exactly one of"):
            sparsejac.jacfwd(lambda x: x)

    def test_jacfwd_both_sparsity_and_coloring_raises(self):
        sparsity = self._diagonal_sparsity()
        coloring = sparsejac.jacfwd_coloring(sparsity)
        with self.assertRaisesRegex(ValueError, "not both"):
            sparsejac.jacfwd(lambda x: x, sparsity, coloring=coloring)

    def test_jacfwd_coloring_sparsity_shape_validation(self):
        with self.assertRaisesRegex(ValueError, "`sparsity` must be rank-2"):
            invalid_sparsity = jsparse.BCOO.fromdense(jnp.ones((5, 5, 5)))
            sparsejac.jacfwd_coloring(invalid_sparsity)

    def test_jacfwd_coloring_accepts_scipy_sparse(self):
        import scipy.sparse as ssparse

        sparsity_scipy = ssparse.eye(_SIZE, format="coo")
        coloring = sparsejac.jacfwd_coloring(sparsity_scipy)
        self.assertIsInstance(coloring, sparsejac.ForwardModeColoring)
        self.assertEqual(coloring.input_coloring.shape, (_SIZE,))

    def test_jacfwd_with_scipy_coloring_matches_jax_coloring(self):
        import scipy.sparse as ssparse

        fn = lambda x: x**2
        sparsity_bcoo = self._diagonal_sparsity()
        sparsity_scipy = ssparse.eye(_SIZE, format="coo")
        x = jax.random.uniform(jax.random.PRNGKey(0), shape=(_SIZE,))
        result_bcoo = sparsejac.jacfwd(fn, coloring=sparsejac.jacfwd_coloring(sparsity_bcoo))(x)
        result_scipy = sparsejac.jacfwd(fn, coloring=sparsejac.jacfwd_coloring(sparsity_scipy))(x)
        onp.testing.assert_array_equal(result_bcoo.todense(), result_scipy.todense())


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
