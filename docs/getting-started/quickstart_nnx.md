# Quick Start — Flax NNX

This page walks through a minimal end-to-end training loop with `NNXTrainer`.

## 1. Install

```bash
pip install "trainax[flax]"
```

## 2. Full example

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from trainax import NNXTrainer, JaxLoader, StepOutput, ValStepOutput, EpochLogger

# ── Data ──────────────────────────────────────────────────────────────────────
x = np.random.randn(1000, 10).astype(np.float32)
y = (x @ np.random.randn(10, 1)).astype(np.float32)

trainloader = JaxLoader({"x": x[:800], "y": y[:800]}, batch_size=32)
valloader   = JaxLoader({"x": x[800:], "y": y[800:]}, batch_size=32)

# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(10, 1, rngs=rngs)

    def __call__(self, x):
        return self.linear(x).squeeze(-1)

model = MLP(nnx.Rngs(0))

# ── Step functions ────────────────────────────────────────────────────────────
def train_step(model, batch):
    def loss_fn(m):
        yhat = jax.vmap(m)(batch["x"])
        return jnp.mean((yhat - batch["y"].squeeze(-1)) ** 2), yhat
    (loss, yhat), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    return StepOutput(loss=loss, y=batch["y"].squeeze(-1), yhat=yhat, gradients=grads)

def val_step(model, batch):
    yhat = jax.vmap(model)(batch["x"])
    loss = jnp.mean((yhat - batch["y"].squeeze(-1)) ** 2)
    return ValStepOutput(loss=loss, y=batch["y"].squeeze(-1), yhat=yhat)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = NNXTrainer(n_epochs=20, callbacks=[EpochLogger("logger")], val_every=5)

model, _ = trainer.train(
    model,
    optax.adam(1e-3),
    train_step,
    trainloader,
    val_step,
    valloader,
)
```

## Key points

- NNX models are **stateful** — `train_step` mutates the model in place; you do not need to pass `train_state=`.
- `nnx.value_and_grad` computes gradients with respect to `nnx.Param` leaves.
- The `NNXTrainer` wraps your step function with `nnx.jit` automatically.
- `model.eval()` / `model.train()` are called internally before/after validation.

See the [NNXTrainer API reference](../api/trainers.md) for all constructor arguments.
