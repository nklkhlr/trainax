# Architecture Overview

## High-level picture

```
User code
  │
  ├─ JaxLoader(data, batch_size, sharding)
  │     yields shuffled batches each epoch
  │
  ├─ train_step / val_step (user-defined callables)
  │     (model, batch) → StepOutput / ValStepOutput
  │
  └─ EQXTrainer / NNXTrainer
        │
        ├─ wraps train_step with eqx.filter_jit / nnx.jit
        ├─ calls _optim_init → opt_state
        ├─ training loop:
        │     for epoch in range(n_epochs):
        │       invoke_callbacks("epoch_start")
        │       for batch in trainloader:
        │         step_fun(model, batch, opt_state, state)
        │         invoke_callbacks("train_step_end")
        │       [if val epoch]:
        │         invoke_callbacks("val_start")
        │         for batch in valloader:
        │           val_step_fun(inference_model, batch)
        │         invoke_callbacks("val_end")
        │       invoke_callbacks("epoch_end")
        │       [optionally save trainer state]
        │     invoke_callbacks("train_end")
        └─ returns TrainOutput(model, state)
```

## Base class vs concrete trainers

`Trainer` (abstract base class) holds:

- The training loop, callback dispatch, sharding setup, and pickle serialization.
- Abstract methods: `_optim_init`, `_setup_step_fun`, `_inference_mode`, `_train_mode`.

Two concrete subclasses override these for each backend:

| | `EQXTrainer` | `NNXTrainer` |
|---|---|---|
| JIT | `eqx.filter_jit` | `nnx.jit` |
| Optimizer | `optax` (external) | `nnx.Optimizer` (internal) |
| State | `eqx.nn.State` pytree | Managed by NNX module |
| Inference mode | `eqx.nn.inference_mode(model)` | `model.eval()` |

## Key design decisions

- **User-defined step functions** — the trainer never knows about the model architecture. It only calls the step function you provide and applies the returned gradients.
- **StepOutput as the contract** — your step function must return a `StepOutput` with `gradients` set; the trainer applies the optax update internally. This keeps the step function pure and testable.
- **Callbacks as observers** — callbacks receive read-only access to epoch results and open file handles. They cannot interrupt the loop (except by exhausting the progress bar — see `EarlyStopping` in the custom callbacks tutorial).
