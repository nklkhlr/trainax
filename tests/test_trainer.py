from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from trainax._callbacks import Callback
from trainax._dataloader import JaxLoader
from trainax._trainer import Trainer
from trainax._types import StepOutput, ValStepOutput

jax.config.update("jax_num_cpu_devices", 2)


def clone_state(state: eqx.nn.State) -> eqx.nn.State:
    leaves, treedef = jax.tree_util.tree_flatten(state)
    cloned_leaves = [jnp.array(leaf) for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, cloned_leaves)


class RecordingCallback(Callback):
    def __init__(self):
        super().__init__("RecordingCallback")
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float | None] = []

    def on_epoch_end(self, model, pbar, epoch, epoch_output, file_handler):
        self.epochs.append(epoch)
        self.train_losses.append(epoch_output.train_loss)
        self.val_losses.append(epoch_output.val_loss)


class CounterCallback(Callback):
    def __init__(self):
        super().__init__("CounterCallback")
        self.on_epoch_start_calls = 0
        self.on_epoch_end_calls = 0
        self.on_step_end_calls = 0
        self.on_val_start_calls = 0
        self.on_val_end_calls = 0

    def on_epoch_start(self, **kwargs):
        self.on_epoch_start_calls += 1

    def on_epoch_end(self, **kwargs):
        self.on_epoch_end_calls += 1

    def on_step_end(self, **kwargs):
        self.on_step_end_calls += 1

    def on_val_start(self, **kwargs):
        self.on_val_start_calls += 1

    def on_val_end(self, **kwargs):
        self.on_val_end_calls += 1


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


def stateless_train_step(
    model: eqx.Module,
    batch: dict[str, jax.Array],
    state=None,
) -> StepOutput:
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


def stateless_val_step(
    model: eqx.Module,
    batch: dict[str, jax.Array],
    state=None,
) -> ValStepOutput:
    preds = jax.vmap(model)(batch["x"])
    loss = jnp.mean((preds - batch["y"]) ** 2)
    return ValStepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
    )


def stateful_train_step(
    model: StatefulNet,
    batch: dict[str, jax.Array],
    state: eqx.nn.State | None,
) -> StepOutput:
    if state is None:
        raise ValueError("Stateful training requires an initial state.")

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


def stateful_val_step(
    model: StatefulNet,
    batch: dict[str, jax.Array],
    state: eqx.nn.State | None,
) -> ValStepOutput:
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


@pytest.fixture
def toy_data():
    weights = np.array([[1.5], [-2.0]], dtype=np.float32)
    bias = np.array([0.1], dtype=np.float32)

    def _make_dataset(n_samples: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n_samples, 2)).astype(np.float32)
        y = x @ weights + bias
        return {"x": x, "y": y}

    return {
        "train": _make_dataset(32, seed=0),
        "val": _make_dataset(16, seed=1),
    }


@pytest.fixture
def train_loader(toy_data):
    return JaxLoader(toy_data["train"], batch_size=len(toy_data["train"]["x"]))


@pytest.fixture
def val_loader(toy_data):
    return JaxLoader(toy_data["val"], batch_size=len(toy_data["val"]["x"]))


@pytest.fixture
def stateless_model():
    return eqx.nn.Linear(2, 1, key=jax.random.PRNGKey(0))


@pytest.fixture
def stateful_model_and_state():
    init_fn = eqx.nn.make_with_state(StatefulNet)
    model, state = init_fn(in_size=2, out_size=1, key=jax.random.PRNGKey(1))
    return model, state


@pytest.fixture
def optimizer():
    return optax.sgd(learning_rate=0.2)


def compute_stateless_loss(model, loader):
    batch = loader._data  # noqa: SLF001
    preds = jax.vmap(model)(batch["x"])
    loss = jnp.mean((preds - batch["y"]) ** 2)
    return float(loss)


def compute_stateful_loss(model, state, loader):
    batch = loader._data  # noqa: SLF001
    preds, _ = model(batch["x"], state, inference=True)
    loss = jnp.mean((preds - batch["y"]) ** 2)
    return float(loss)


def test_trainer_requires_validation_step(
    stateless_model,
    train_loader,
    val_loader,
    optimizer,
):
    trainer = Trainer(
        n_epochs=1,
        callbacks=[],
        val_every=1,
        use_rich=False,
    )
    with pytest.raises(ValueError, match="val_step is not defined"):
        trainer.train(
            model=stateless_model,
            optim=optimizer,
            train_step=stateless_train_step,
            trainloader=train_loader,
            val_step=None,
            valloader=val_loader,
            jit_fun=lambda fn: fn,
        )


@pytest.mark.parametrize("data_sharding", [None, [0, 1]])
def test_trainer_trains_stateless_model(
    stateless_model, train_loader, optimizer, data_sharding
):
    print(f"\n\nSharding stateless: {data_sharding}\n")

    trainer = Trainer(
        n_epochs=4,
        callbacks=[],
        val_every=1,
        use_rich=False,
        data_sharding=data_sharding,
    )
    initial_loss = compute_stateless_loss(stateless_model, train_loader)

    trained_model, trained_state = trainer.train(
        model=stateless_model,
        optim=optimizer,
        train_step=stateless_train_step,
        trainloader=train_loader,
        val_step=None,
        valloader=None,
        jit_fun=lambda fn: fn,
    )

    final_loss = compute_stateless_loss(trained_model, train_loader)
    assert final_loss < initial_loss
    assert trained_state is None


@pytest.mark.parametrize("data_sharding", [None, [0, 1]])
def test_trainer_trains_stateful_model_and_updates_state(
    stateful_model_and_state, train_loader, optimizer, data_sharding
):
    print(f"\n\nSharding stateful: {data_sharding}\n")

    model, base_state = stateful_model_and_state
    train_state = clone_state(base_state)
    trainer = Trainer(
        n_epochs=3,
        callbacks=[],
        val_every=1,
        use_rich=False,
        data_sharding=data_sharding,
    )

    initial_loss = compute_stateful_loss(
        model,
        clone_state(train_state),
        train_loader,
    )

    trained_model, trained_state = trainer.train(
        model=model,
        optim=optimizer,
        train_step=stateful_train_step,
        trainloader=train_loader,
        val_step=None,
        valloader=None,
        train_state=train_state,
        jit_fun=lambda fn: fn,
    )

    assert trained_state is not None
    assert trained_state is not train_state
    final_loss = compute_stateful_loss(
        trained_model,
        clone_state(trained_state),
        train_loader,
    )
    assert final_loss < initial_loss
    clone_state(trained_state)


@pytest.mark.filterwarnings("ignore:rich ")
def test_trainer_rich_progress(
    stateless_model,
    train_loader,
    optimizer,
):
    trainer = Trainer(
        n_epochs=4,
        callbacks=[],
        val_every=1,
        use_rich=True,
    ).train(
        model=stateless_model,
        optim=optimizer,
        train_step=stateless_train_step,
        trainloader=train_loader,
        val_step=None,
        valloader=None,
        jit_fun=lambda fn: fn,
    )


def test_trainer_validation_frequency_and_recording(
    stateless_model,
    train_loader,
    val_loader,
    optimizer,
):
    val_calls: list[float] = []

    def tracking_val_step(_model, batch):
        preds = jax.vmap(stateless_model)(batch["x"])
        loss = jnp.mean((preds - batch["y"]) ** 2)
        val_calls.append(float(loss))
        return ValStepOutput(loss=loss, y=batch["y"], yhat=preds)

    callback = RecordingCallback()
    trainer = Trainer(
        n_epochs=3,
        callbacks=[callback],
        val_every=2,
        use_rich=False,
    )

    trainer.train(
        model=stateless_model,
        optim=optimizer,
        train_step=stateless_train_step,
        trainloader=train_loader,
        val_step=tracking_val_step,
        valloader=val_loader,
        jit_fun=lambda fn: fn,
    )

    assert len(val_calls) == len(val_loader)
    assert callback.val_losses.count(None) == 2
    assert callback.val_losses.count(None) + 1 == trainer.n_epochs


def test_trainer_invokes_callbacks(
    stateless_model,
    train_loader,
    optimizer,
):
    callback = CounterCallback()
    trainer = Trainer(
        n_epochs=2,
        callbacks=[callback],
        val_every=1,
        use_rich=False,
    )

    trainer.train(
        model=stateless_model,
        optim=optimizer,
        train_step=stateless_train_step,
        trainloader=train_loader,
        val_step=None,
        valloader=None,
        jit_fun=lambda fn: fn,
    )

    assert callback.on_epoch_start_calls == trainer.n_epochs
    assert callback.on_epoch_end_calls == trainer.n_epochs
    assert callback.on_step_end_calls == trainer.n_epochs * len(train_loader)
    assert callback.on_val_start_calls == 0
    assert callback.on_val_end_calls == 0


def test_trainer_stateful_requires_initial_state(
    stateful_model_and_state,
    train_loader,
    optimizer,
):
    model, _ = stateful_model_and_state
    trainer = Trainer(
        n_epochs=1,
        callbacks=[],
        val_every=1,
        use_rich=False,
    )

    with pytest.raises(
        TypeError, match="missing 1 required positional argument"
    ):
        trainer.train(
            model=model,
            optim=optimizer,
            train_step=stateful_train_step,
            trainloader=train_loader,
            val_step=None,
            valloader=None,
            train_state=None,
            jit_fun=lambda fn: fn,
        )


def test_trainer_set_aggregate_steps():
    trainer = Trainer(
        n_epochs=1,
        callbacks=[],
        val_every=1,
        use_rich=False,
    )
    trainer.set_aggregate_steps("max")
    assert trainer.aggregate_steps == "max"

    with pytest.raises(ValueError, match="Invalid aggregate_steps"):
        trainer.set_aggregate_steps("median")  # type: ignore[arg-type]
