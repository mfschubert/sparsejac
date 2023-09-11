# sparsejac 0.0.0
Efficient forward- and reverse-mode sparse Jacobians using Jax.

Sparse Jacobians are frequently encountered in the simulation of physical systems. Jax tranformations `jacfwd` and `jacrev` make it easy to compute dense Jacobians, but these are wasteful when the Jacobian is sparse. `sparsejac` provides a function to more efficiently compute the Jacobian if its sparsity is known. It makes use of the recently-introduced `jax.experimental.sparse` module.

## Install
```
pip install sparsejac
```

## Example
A trivial example with a diagonal Jacobian follows:

```python
fn = lambda x: x**2
sparsity = jax.experimental.sparse.BCOO.fromdense(jnp.eye(10000))
x = jax.random.uniform(jax.random.PRNGKey(0), shape=(10000,))

sparse_fn = jax.jit(sparsejac.jacrev(fn, sparsity))
dense_fn = jax.jit(jax.jacrev(fn))

assert jnp.all(sparse_fn(x).todense() == dense_fn(x))

%timeit sparse_fn(x).block_until_ready()
%timeit dense_fn(x).block_until_ready()
```

And, the performance improvement can easily be seen:

```
10000 loops, best of 5: 96.5 µs per loop
10 loops, best of 5: 56.9 ms per loop
```
