from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from trainax._callbacks import Callback
from trainax._dataloader import JaxLoader
from trainax._trainer import Trainer
from trainax._types import StepOutput, ValStepOutput


class TrackingCallback(Callback):
    def __init__(self, name: str):
        super().__init__(name)
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float | None] = []

    def on_epoch_end(self, model, pbar, epoch, epoch_output, file_handler):
        self.epochs.append(epoch)
        self.train_losses.append(float(epoch_output.train_loss))
        val_loss = epoch_output.val_loss
        self.val_losses.append(None if val_loss is None else float(val_loss))


class StatefulNet(eqx.Module):
    linear: eqx.nn.Linear
    norm: eqx.nn.BatchNorm

    def __init__(self, in_size: int, out_size: int, *, key: jax.Array):
        key_linear, _ = jax.random.split(key)
        self.linear = eqx.nn.Linear(in_size, out_size, key=key_linear)
        self.norm = eqx.nn.BatchNorm(
            input_size=out_size,
            axis_name=("batch",),
            inference=False,
            mode="batch",
        )

    def __call__(
        self, inputs: jax.Array, state: eqx.nn.State, *args, **kwargs
    ) -> tuple[jax.Array, eqx.nn.State]:
        linear_out = self.linear(inputs)
        outputs, state = self.norm(
            linear_out,
            state,
        )
        return outputs, state


def loss_fn(yhat, y):
    return jnp.mean((yhat - y) ** 2)


def stateless_train_step(model: eqx.Module, batch) -> StepOutput:
    def _step(m: eqx.Module):
        preds = jax.vmap(m, axis_name="batch")(batch["x"])
        loss = loss_fn(preds, batch["y"])
        return loss, preds

    (loss, preds), grads = eqx.filter_value_and_grad(
        _step,
        has_aux=True,
    )(model)
    return StepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
        gradients=grads,
    )


def val_step(model: eqx.Module, batch) -> ValStepOutput:
    preds = jax.vmap(model, axis_name="batch")(batch["x"])
    # to handle stateful and stateless models in the same val step
    if isinstance(preds, tuple):
        preds = preds[0]

    loss = loss_fn(preds, batch["y"])
    return ValStepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
    )


def stateful_train_step(model: StatefulNet, batch, state) -> StepOutput:
    if state is None:
        raise ValueError("Stateful training requires a state.")

    def _step(m: StatefulNet, state_: eqx.nn.State):
        preds, new_state = jax.vmap(m, axis_name="batch", in_axes=(0, None))(
            batch["x"], state_
        )
        loss = loss_fn(preds, batch["y"])
        return loss, (preds, new_state)

    (loss, (preds, new_state)), grads = eqx.filter_value_and_grad(
        _step,
        has_aux=True,
    )(model, state)

    return StepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
        gradients=grads,
        state=new_state,
    )


def build_loaders() -> tuple[JaxLoader, JaxLoader]:
    weights = np.array([[1.5], [-2.0]], dtype=np.float32)
    bias = np.array([0.1], dtype=np.float32)
    rng = np.random.default_rng(0)

    def _dataset(n_samples: int) -> dict[str, np.ndarray]:
        x = rng.normal(size=(n_samples, 2)).astype(np.float32)
        y = x @ weights + bias
        return {"x": x, "y": y}

    train_loader = JaxLoader(_dataset(128), batch_size=32)
    val_loader = JaxLoader(_dataset(64), batch_size=32)
    return train_loader, val_loader


def to_series(values: list[float | None]) -> list[float]:
    series: list[float] = []
    last = np.nan
    for value in values:
        if value is None:
            series.append(last)
        else:
            last = value
            series.append(value)
    return series


def run_training():
    train_loader, val_loader = build_loaders()
    optimizer = optax.adam(learning_rate=0.05)

    # stateless model
    stateless_cb = TrackingCallback("stateless")
    stateless_trainer = Trainer(
        n_epochs=15,
        callbacks=[stateless_cb],
        val_every=1,
        use_rich=False,
    )
    stateless_model = eqx.nn.Linear(2, 1, key=jax.random.PRNGKey(0))
    stateless_trainer.train(
        model=stateless_model,
        optim=optimizer,
        train_step=stateless_train_step,
        trainloader=train_loader,
        val_step=val_step,
        valloader=val_loader,
    )

    # stateful model
    stateful_cb = TrackingCallback("stateful")
    stateful_trainer = Trainer(
        n_epochs=15,
        callbacks=[stateful_cb],
        val_every=1,
        use_rich=False,
    )
    init_fn = eqx.nn.make_with_state(StatefulNet)
    stateful_model, state = init_fn(
        in_size=2,
        out_size=1,
        key=jax.random.PRNGKey(1),
    )
    stateful_trainer.train(
        model=stateful_model,
        optim=optimizer,
        train_step=stateful_train_step,
        trainloader=train_loader,
        val_step=val_step,
        valloader=val_loader,
        train_state=state,
    )

    return stateless_cb, stateful_cb


def main():
    stateless_cb, stateful_cb = run_training()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    epochs = [e + 1 for e in stateless_cb.epochs]

    axes[0].plot(epochs, stateless_cb.train_losses, label="train")
    axes[0].plot(epochs, to_series(stateless_cb.val_losses), label="val")
    axes[0].set_title("Stateless model")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend()

    axes[1].plot(epochs, stateful_cb.train_losses, label="train")
    axes[1].plot(epochs, to_series(stateful_cb.val_losses), label="val")
    axes[1].set_title("Stateful model")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle("Training convergence for stateless vs. stateful models")
    fig.tight_layout()
    plt.savefig("train_small_models.pdf")
    plt.savefig("train_small_models.svg")


if __name__ == "__main__":
    main()
