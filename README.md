# sparsejac: Efficient sparse Jacobians using Jax
`v0.2.0`

Sparse Jacobians are frequently encountered in the simulation of physical systems. Jax tranformations `jacfwd` and `jacrev` make it easy to compute dense Jacobians, but these are wasteful when the Jacobian is sparse. `sparsejac` provides a function to more efficiently compute the Jacobian if its sparsity is known. It makes use of the recently-introduced `jax.experimental.sparse` module.

## Install
```
pip install sparsejac
```

## Example
A trivial example with a diagonal Jacobian follows:

```python
fn = lambda x: x**2
x = jax.random.uniform(jax.random.PRNGKey(0), shape=(10000,))

@jax.jit
def sparse_jacrev_fn(x):
  with jax.ensure_compile_time_eval():
    sparsity = jax.experimental.sparse.BCOO.fromdense(jnp.eye(10000))
    jacrev_fn = sparsejac.jacrev(fn, sparsity=sparsity)
  return jacrev_fn(x)

dense_jacrev_fn = jax.jit(jax.jacrev(fn))

assert jnp.all(sparse_jacrev_fn(x).todense() == dense_jacrev_fn(x))

%timeit sparse_jacrev_fn(x).block_until_ready()
%timeit dense_jacrev_fn(x).block_until_ready()
```

And, the performance improvement can easily be seen:

```
93.1 µs ± 17.2 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
182 ms ± 26.9 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Performance

- In general, it is preferable to directly provide the sparsity, rather than obtaining it from a dense matrix.
- GPU may show minimal or no performance advantage over CPU.
- Users are encouraged to test `jacrev` and `jacfwd` on their specific problem to select the most performant option.
