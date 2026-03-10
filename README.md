<div align="left">
<img src="https://raw.githubusercontent.com/nklkhlr/trainax/main/media/logo.svg" alt="logo"></img>
</div>

# Ligthweigt training utilities for JAX models

> **Work in progress.** This library is under active development. APIs may change without notice and no stability guarantees are made.

`trainax` is a lightweight training utility library for [JAX](https://docs.jax.dev/en/latest/index.html) models. 
It provides a shared data loader for [equinox](https://github.com/patrick-kidger/equinox) models, 
including stateful models, and for [flax NNX](https://flax.readthedocs.io/en/latest/nnx/) models and handles model- and 
data-sharding as well ass just-in-time compilation for both libraries 
automatically.
Additionally, `trainax` provides a `Trainer` class that can easily be customized
via callbacks (similar but much less comprehensive to the use of callbacks
in [`pytorch-lighting`](https://github.com/Lightning-AI/pytorch-lightning))

## Installation

**Core package** (no model backend):
```bash
pip install trainax
```

**With Equinox backend:**
```bash
pip install "trainax[eqx]"
```

**With flax NNX backend:**
```bash
pip install "trainax[flax]"
```

**Using uv:**
```bash
uv add trainax
uv sync --group eqx   # or --group flax, or both
```

## Usage

### Equinox

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from trainax import EQXTrainer, JaxLoader, StepOutput, ValStepOutput
from trainax import EpochLogger, BestModelSaver

# Data
x = np.random.randn(1000, 10).astype(np.float32)
y = (x @ np.random.randn(10, 1)).astype(np.float32).squeeze(-1)
trainloader = JaxLoader({"x": x[:800], "y": y[:800]}, batch_size=32)
valloader   = JaxLoader({"x": x[800:], "y": y[800:]}, batch_size=32)

# Model
model = eqx.nn.MLP(10, 1, width_size=32, depth=2, key=jax.random.key(0))

# Step functions
def train_step(model, batch):
    def loss_fn(m):
        yhat = jax.vmap(m)(batch["x"]).squeeze(-1)
        return jnp.mean((yhat - batch["y"]) ** 2), yhat
    (loss, yhat), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    return StepOutput(loss=loss, y=batch["y"], yhat=yhat, gradients=grads)

def val_step(model, batch):
    yhat = jax.vmap(model)(batch["x"]).squeeze(-1)
    loss = jnp.mean((yhat - batch["y"]) ** 2)
    return ValStepOutput(loss=loss, y=batch["y"], yhat=yhat)

# Train
def save_fn(m, epoch): eqx.tree_serialise_leaves("best.eqx", m)

trainer = EQXTrainer(
    n_epochs=50,
    callbacks=[
        EpochLogger("logger"),
        BestModelSaver(save_fn, key="val_loss"),
    ],
    val_every=5,
)

model, _ = trainer.train(
    model, optax.adam(1e-3), train_step, trainloader, val_step, valloader
)
```

### Flax NNX

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from trainax import NNXTrainer, JaxLoader, StepOutput, ValStepOutput, EpochLogger

# Model
class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(10, 1, rngs=rngs)
    def __call__(self, x):
        return self.linear(x).squeeze(-1)

model = MLP(nnx.Rngs(0))

# Step functions
def train_step(model, batch):
    def loss_fn(m):
        yhat = jax.vmap(m)(batch["x"])
        return jnp.mean((yhat - batch["y"]) ** 2), yhat
    (loss, yhat), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    return StepOutput(loss=loss, y=batch["y"], yhat=yhat, gradients=grads)

def val_step(model, batch):
    yhat = jax.vmap(model)(batch["x"])
    loss = jnp.mean((yhat - batch["y"]) ** 2)
    return ValStepOutput(loss=loss, y=batch["y"], yhat=yhat)

# Train
trainer = NNXTrainer(n_epochs=50, callbacks=[EpochLogger("logger")], val_every=5)
model, _ = trainer.train(
    model, optax.adam(1e-3), train_step, trainloader, val_step, valloader
)
```

## Development

```bash
# Install all dependency groups
uv sync --all-groups

# Run tests
uv run pytest

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Build docs
uv sync --group docs
uv run mkdocs serve
```

## Disclaimer

This project is a work in progress. It is developed for personal research use and shared as-is. Breaking changes may occur at any time, tests may be incomplete, and there are no guarantees of correctness, stability, or support.
