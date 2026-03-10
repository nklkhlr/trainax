# Sharding

`trainax` supports **data-parallel sharding** via JAX's `jax.sharding` API. Model sharding is a placeholder for future work.

## Configuring sharding

Pass `data_sharding=` to the trainer constructor. Accepted formats:

| Format | Example | Meaning |
|---|---|---|
| `int` | `data_sharding=4` | Use first 4 devices from `jax.devices()` |
| `list[int]` | `data_sharding=[0, 2]` | Use devices at indices 0 and 2 |
| `list[jax.Device]` | `data_sharding=jax.devices()[:2]` | Use explicit device objects |

You can also set sharding after construction:

```python
trainer.set_sharding({"data": 4, "model": None})
# or equivalently:
trainer.sharding = {"data": 4, "model": None}
```

Passing `None` disables sharding entirely.

## How it works internally

1. The trainer calls `_set_sharding((data_devs, model_devs))`.
2. A `jax.sharding.Mesh` is created with the requested devices along the `"data"` axis.
3. A `jsd.NamedSharding(mesh, PartitionSpec("data"))` is stored as `trainer._sharding["data"]`.
4. During training, `trainloader.set_sharding(data_sharding)` is called, which configures `JaxLoader` to use `jax.make_array_from_process_local_data` for each batch.
5. Each batch's leading axis is split across all devices — a 32-sample batch on 4 devices means each device receives 8 samples.

## Constraints

- The **batch size must be divisible** by the number of shards (`JaxLoader` asserts this at sharding setup time).
- For `SingleBatchJaxLoader`, the dataset size doesn't need to be divisible; the loader trims the remainder and emits a `UserWarning`.
- Step functions do **not** need to change — JAX's compiler handles the distribution transparently under `jit`.

## Local testing with simulated devices

```python
import jax
# Must be called before any JAX operation
jax.config.update("jax_num_cpu_devices", 4)

import jax.numpy as jnp  # import after config update
```

This creates 4 virtual CPU devices, letting you test sharding code locally without a GPU/TPU cluster.

## Verifying sharding

After training, inspect a batch from the loader:

```python
for batch in trainloader:
    print(batch["x"].sharding)
    print([s.data.shape for s in batch["x"].addressable_shards])
    break
```

Each shard should have shape `(batch_size // n_devices, ...)`.
