from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from flax import nnx

from trainax._dataloader import JaxLoader
from trainax._trainer import NNXTrainer
from trainax._types import StepOutput, ValStepOutput

jax.config.update("jax_num_cpu_devices", 2)


N_DIMS = 2
N_HIDDEN = 2
N_OUT = 1


class SimpleModel(nnx.Module):
    def __init__(
        self, in_size: int, hidden_size: int, out_size: int, rngs: nnx.Rngs
    ):
        super().__init__()
        self.linear1 = nnx.Linear(in_size, hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, out_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear1(x)
        x = jax.nn.relu(x)
        return self.linear2(x)


class BatchNormModel(nnx.Module):
    def __init__(
        self, in_size: int, hidden_size: int, out_size: int, rngs: nnx.Rngs
    ):
        super().__init__()
        self.linear1 = nnx.Linear(in_size, hidden_size, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_size, out_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = jax.nn.relu(x)
        return self.linear2(x)


@pytest.fixture
def toy_data():
    weights = np.array([[1.5], [-2.0]], dtype=np.float32)
    bias = np.array([0.1], dtype=np.float32)

    def _make_dataset(n_samples: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n_samples, N_DIMS)).astype(np.float32)
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


def simple_model():
    return SimpleModel(
        in_size=N_DIMS, hidden_size=N_HIDDEN, out_size=N_OUT, rngs=nnx.Rngs(42)
    )


def batch_norm_model():
    return BatchNormModel(
        in_size=N_DIMS, hidden_size=N_HIDDEN, out_size=N_OUT, rngs=nnx.Rngs(42)
    )


@pytest.fixture
def optimizer():
    return optax.adam(learning_rate=1e-3)


def loss_fn(model, x, y):
    preds = model(x)
    return jnp.mean((preds - y) ** 2), preds


def compute_loss(model, loader):
    loss, _ = loss_fn(model, loader._data["x"], loader._data["y"])
    return loss


def train_step(model, batch):
    (loss, preds), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, batch["x"], batch["y"]
    )
    return StepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
        gradients=grads,
    )


def val_step(model, batch):
    loss, preds = loss_fn(model, batch["x"], batch["y"])
    return ValStepOutput(
        loss=loss,
        y=batch["y"],
        yhat=preds,
    )


@pytest.mark.parametrize("model_init", [simple_model, batch_norm_model])
@pytest.mark.parametrize("data_sharding", [None, [0, 1]])
def test_nnx_simple_model(
    model_init,
    train_loader,
    val_loader,
    optimizer,
    data_sharding,
):
    print(f"\n\nSharding stateless: {data_sharding} for model {model_init}\n")

    model = model_init()
    model.eval()
    trainer = NNXTrainer(
        n_epochs=1,
        callbacks=[],
        val_every=1,
        use_rich=False,
    )
    initial_loss = compute_loss(model, train_loader)
    model.train()

    trained_model, trained_state = trainer.train(
        model=model,
        optim=optimizer,
        train_step=train_step,
        trainloader=train_loader,
        val_step=val_step,
        valloader=val_loader,
    )

    trained_model.eval()
    final_loss = compute_loss(trained_model, train_loader)
    assert final_loss < initial_loss
    assert trained_state is None
