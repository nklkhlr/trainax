# trainax

`trainax` is a lightweight training utility library for JAX models. It provides a shared data loader and two concrete trainer backends — one for [Equinox](https://github.com/patrick-kidger/equinox) functional models, and one for [Flax NNX](https://flax.readthedocs.io/en/latest/nnx/) stateful models.

## Features

- **Two trainer backends** — `EQXTrainer` for Equinox/optax workflows and `NNXTrainer` for Flax NNX.
- **Flexible data loading** — `JaxLoader` handles shuffling, batching, and optional JAX sharding for any numpy/jax dict dataset.
- **Callback system** — built-in callbacks for progress bars, loss tracking, metric logging, and best-model checkpointing; easily extensible.
- **Stateful model support** — `EQXTrainer` threads `eqx.nn.State` through step functions across epochs.
- **Multi-device sharding** — configure data and model sharding in one line via `data_sharding=` / `model_sharding=`.
- **Serialization** — save and restore trainer state mid-training via `epoch_state_file=`.

## Installation

```bash
# Core (no backend)
pip install trainax

# With Equinox backend
pip install "trainax[eqx]"

# With Flax NNX backend
pip install "trainax[flax]"

# Both backends
pip install "trainax[eqx,flax]"
```

With `uv`:
```bash
uv add trainax
uv sync --group eqx --group flax
```

## Quick example

```python
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from trainax import EQXTrainer, JaxLoader, StepOutput, EpochLogger

# Build a tiny model
model = eqx.nn.Linear(10, 1, key=jax.random.key(0))

# Wrap data in a loader
loader = JaxLoader({"x": x_train, "y": y_train}, batch_size=32)

# Define a training step
def train_step(model, batch):
    def loss_fn(m):
        yhat = jax.vmap(m)(batch["x"])
        return jnp.mean((yhat - batch["y"]) ** 2), yhat
    (loss, yhat), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    return StepOutput(loss=loss, y=batch["y"], yhat=yhat, gradients=grads)

trainer = EQXTrainer(n_epochs=50, callbacks=[EpochLogger("logger")])
model, _ = trainer.train(
    model, optax.adam(1e-3), train_step, loader, val_step=None, valloader=None
)
```

## Links

- [Getting Started](getting-started/installation.md)
- [API Reference](api/trainers.md)
- [Tutorials](tutorials/eqx_regression.ipynb)
- [Source on GitHub](https://github.com/nikolaik/jax-trainer)
