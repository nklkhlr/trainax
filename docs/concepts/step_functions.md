# Step Functions

Step functions are the core user-provided building blocks. The trainer wraps them with JIT and calls them inside the training loop.

## Stateless step function (EQX)

```python
def train_step(model, batch) -> StepOutput:
    ...
```

Use this when your model has no mutable state (no `BatchNorm`, etc.).

## Stateful step function (EQX only)

```python
def train_step(model, batch, state) -> StepOutput:
    ...
```

Use this when your model has `eqx.nn.State` (e.g. `BatchNorm`). The trainer detects which signature to use at runtime by checking whether `train_state` was passed to `.train()`.

The returned `StepOutput` **must** have `out.state` set to the updated state. The trainer extracts it via `out.state` and threads it to the next step.

## NNX step function

```python
def train_step(model: nnx.Module, batch) -> StepOutput:
    ...
```

NNX models are stateful objects — they mutate in place. There is no separate state argument. The trainer passes the same module object to every step.

## Validation step functions

Validation step functions use the same signatures but return `ValStepOutput` (no `gradients` field):

```python
def val_step(model, batch) -> ValStepOutput:
    ...
```

For stateful EQX models, `EQXTrainer` wraps the model with `eqx.Partial(inference_model, state=state)` before validation, so the val_step does not receive a `state` argument.

## Required fields in StepOutput

| Field | Required | Description |
|---|---|---|
| `loss` | Yes | Scalar loss value |
| `y` | Yes | Ground truth targets for the batch |
| `yhat` | Yes | Model predictions |
| `gradients` | **Yes (train)** | Parameter gradients; trainer applies the update |
| `state` | Stateful only | Updated mutable state pytree |
| `metrics` | No | Optional dict of extra scalars |

`gradients` must not be `None` in training step outputs — the trainer raises `ValueError` if it is.

## Example: EQX stateless train step

```python
def train_step(model, batch):
    def loss_fn(m):
        yhat = jax.vmap(m)(batch["x"]).squeeze(-1)
        return jnp.mean((yhat - batch["y"]) ** 2), yhat

    (loss, yhat), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    return StepOutput(loss=loss, y=batch["y"], yhat=yhat, gradients=grads)
```

## Example: NNX train step

```python
def train_step(model, batch):
    def loss_fn(m):
        yhat = jax.vmap(m)(batch["x"]).squeeze(-1)
        return jnp.mean((yhat - batch["y"]) ** 2), yhat

    (loss, yhat), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    return StepOutput(loss=loss, y=batch["y"], yhat=yhat, gradients=grads)
```
