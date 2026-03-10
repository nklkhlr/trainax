# Quick Start — Equinox

This page walks through a minimal end-to-end training loop with `EQXTrainer`.

## 1. Install

```bash
pip install "trainax[eqx]"
```

## 2. Full example

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx

from trainax import EQXTrainer, JaxLoader, StepOutput, EpochLogger, BestModelSaver

# ── Data ──────────────────────────────────────────────────────────────────────
key = jax.random.key(0)
x = np.random.randn(1000, 10).astype(np.float32)
y = (x @ np.random.randn(10, 1)).astype(np.float32)

trainloader = JaxLoader({"x": x[:800], "y": y[:800]}, batch_size=32)
valloader   = JaxLoader({"x": x[800:], "y": y[800:]}, batch_size=32)

# ── Model ─────────────────────────────────────────────────────────────────────
model = eqx.nn.Linear(10, 1, key=key)

# ── Step functions ────────────────────────────────────────────────────────────
def train_step(model, batch):
    def loss_fn(m):
        yhat = jax.vmap(m)(batch["x"]).squeeze(-1)
        return jnp.mean((yhat - batch["y"].squeeze(-1)) ** 2), yhat
    (loss, yhat), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    return StepOutput(loss=loss, y=batch["y"].squeeze(-1), yhat=yhat, gradients=grads)

def val_step(model, batch):
    from trainax import ValStepOutput
    yhat = jax.vmap(model)(batch["x"]).squeeze(-1)
    loss = jnp.mean((yhat - batch["y"].squeeze(-1)) ** 2)
    return ValStepOutput(loss=loss, y=batch["y"].squeeze(-1), yhat=yhat)

# ── Trainer ───────────────────────────────────────────────────────────────────
callbacks = [
    EpochLogger("logger"),
    BestModelSaver("saver", save_path="best_model.pkl", monitor="val_loss"),
]

trainer = EQXTrainer(n_epochs=20, callbacks=callbacks, val_every=5)

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

- `train_step` must return a `StepOutput` with `gradients` set — the trainer applies parameter updates internally via the optax optimiser.
- `val_step` returns a `ValStepOutput` without gradients.
- `BestModelSaver` saves the model whenever `val_loss` improves.
- Set `val_every=N` to run validation every *N* epochs.

See the [EQXTrainer API reference](../api/trainers.md) for all constructor arguments.
