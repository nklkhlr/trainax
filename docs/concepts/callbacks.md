# Callback Hooks

## Hook execution order

```
trainer.train() called
│
├─ file_handler.open()           # opens all continuous_files
│
├─ [epoch 0..n_epochs-1]
│     on_epoch_start(epoch, pbar, file_handler)
│     │
│     ├─ [step 0..n_batches-1]
│     │     on_train_step_end(step_output, pbar)
│     │
│     ├─ [if (epoch+1) % val_every == 0 and valloader is not None]
│     │     on_val_start(epoch, loader)
│     │     [for each val batch]
│     │       (val_step called internally — no per-step callback)
│     │     on_val_end(epoch, data)   # data = list[ValStepOutput]
│     │
│     on_epoch_end(model, pbar, epoch, epoch_output, file_handler)
│
├─ file_handler.close()
└─ on_train_end(pbar)
```

## Validation gating

`on_val_start` and `on_val_end` are only called on epochs where `(epoch + 1) % val_every == 0`. The trainer keeps `val_every` in sync with each callback's own `val_every` attribute.

This matters for `BestModelSaver`: it skips evaluation epochs where validation was not run (when `key="val_loss"`).

## FileHandler integration

`FileHandler` wraps a `dict[str, PathLike]` mapping. During training it holds open writeable file handles. Callbacks access them via `file_handler["my_key"]`.

```python
# Trainer constructor
trainer = EQXTrainer(
    ...
    continuous_files={"train_loss": "train.csv", "val_loss": "val.csv"},
)

# Inside a callback
def on_epoch_end(self, model, pbar, epoch, epoch_output, file_handler):
    file_handler["train_loss"].write(f"{epoch_output.train_loss}\n")
```

The `FileHandler` raises `KeyError` if a key is not registered, with a hint to add the file via `continuous_files`.

## Built-in callbacks

| Class | Hooks used | Purpose |
|---|---|---|
| `EpochLogger` | `on_epoch_end` | Log train/val loss via `logging.Logger` |
| `PbarHandler` | `on_epoch_end`, `on_train_end` | Update tqdm progress bar postfix |
| `LossMetricTracker` | `on_epoch_end` | Write losses to disk, cache in `self.losses` |
| `BestModelSaver` | `on_epoch_end` | Save model when monitored metric improves |
| `NNXBestModelSaver` | `on_epoch_end` | Same, but for NNX (uses `nnx.state()` serialization) |
| `NNXMetricTracker` | `on_epoch_end` | NNX variant of `LossMetricTracker` |

## Writing a custom callback

Subclass `Callback`, assign a unique `name`, and override the hooks you need:

```python
from trainax import Callback

class MyCallback(Callback):
    def __init__(self):
        super().__init__(name="MyCallback")

    def on_epoch_end(self, model, pbar, epoch, epoch_output, file_handler):
        print(f"Epoch {epoch}: train_loss={epoch_output.train_loss:.4f}")
```
