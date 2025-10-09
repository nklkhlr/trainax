from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import optax

import equinox as eqx
import jax
import jax.numpy as jnp

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
        self.val_losses.append(
            None if val_loss is None else float(val_loss)
        )


class StatefulNet(eqx.Module):
    linear: eqx.nn.Linear
    norm: eqx.nn.BatchNorm

    def __init__(self, in_size: int, out_size: int, *, key: jax.Array):
        key_linear, _ = jax.random.split(key)
        self.linear = eqx.nn.Linear(in_size, out_size, key=key_linear)
        self.norm = eqx.nn.BatchNorm(
            input_size=out_size,
            axis_name=(),
            inference=False,
            mode="batch",
        )

    def __call__(
        self,
        inputs: jax.Array,
        state: eqx.nn.State,
        *,
        inference: bool = False,
    ) -> tuple[jax.Array, eqx.nn.State]:
        norm_state = state.substate(self.norm)
        norm_in = jnp.swapaxes(inputs, 0, 1)
        linear_out = self.linear(norm_in)
        norm_out, updated_norm_state = self.norm(
            linear_out,
            norm_state,
            inference=inference,
        )
        outputs = jnp.swapaxes(norm_out, 0, 1)
        updated_state = state.update(updated_norm_state)
        return outputs, updated_state


def stateless_train_step(model: eqx.Module, batch, state=None) -> StepOutput:
    def loss_fn(m: eqx.Module):
        preds = jax.vmap(m)(batch["x"])
        loss = jnp.mean((preds - batch["y"]) ** 2)
        return loss, preds

    (loss, preds), grads = eqx.filter_value_and_grad(
        loss_fn,
        has_aux=True,
    )(model)
    return StepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
        gradients=grads,
    )


def stateless_val_step(model: eqx.Module, batch, state=None) -> ValStepOutput:
    preds = jax.vmap(model)(batch["x"])
    loss = jnp.mean((preds - batch["y"]) ** 2)
    return ValStepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
    )


def stateful_train_step(model: StatefulNet, batch, state) -> StepOutput:
    if state is None:
        raise ValueError("Stateful training requires a state.")

    def loss_fn(m: StatefulNet, state_: eqx.nn.State):
        preds, new_state = m(batch["x"], state_, inference=False)
        loss = jnp.mean((preds - batch["y"]) ** 2)
        return loss, (preds, new_state)

    (loss, (preds, new_state)), grads = eqx.filter_value_and_grad(
        loss_fn,
        has_aux=True,
    )(model, state)

    return StepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
        gradients=grads,
        state=new_state,
    )


def stateful_val_step(model: StatefulNet, batch, state) -> ValStepOutput:
    if state is None:
        raise ValueError("Stateful validation requires a state.")
    preds, new_state = model(batch["x"], state, inference=True)
    loss = jnp.mean((preds - batch["y"]) ** 2)
    return ValStepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
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

    # Stateless model -------------------------------------------------
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
        val_step=stateless_val_step,
        valloader=val_loader,
    )

    # Stateful model --------------------------------------------------
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
        val_step=stateful_val_step,
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
    plt.show()


if __name__ == "__main__":
    main()
