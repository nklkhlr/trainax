from functools import partial

import equinox as eqx
import jax
import jax.sharding as jsd
from jaxtyping import Array
from test_loader import get_test_data

from trainax import JaxLoader, Trainer


def step(model, data: dict[str, Array]) -> dict[str, Array]:
    yhat = jax.vmap(model)(data["x"])
    loss = (yhat - data["y"]) ** 2

    return {"train_loss": loss}


def init_trainer(model: eqx.Module, callbacks: dict, **kwargs):
    step_fun = partial(step, model)

    # TODO: default callbacks?
    if callbacks.get("epoch") is None:
        callbacks["epoch"] = []
    if callbacks.get("step") is None:
        callbacks["step"] = []
    if callbacks.get("logger") is None:
        callbacks["logger"] = []

    return Trainer(
        make_step=step_fun,
        epoch_callbacks=callbacks["epoch"],
        step_callbacks=callbacks["step"],
        logger_callbacks=callbacks["logger"],
        file_handlers=[],
        **kwargs,
    )


class TestTrainer:
    n_train = 100
    n_val = 30
    m = 10
    train_seed = 42
    val_seed = 12

    train_data = JaxLoader(
        get_test_data(train_seed, n_train, m),
        batch_size=50,
    )
    val_data = JaxLoader(
        get_test_data(val_seed, n_val, m),
        batch_size=30,
    )

    data_mesh = jax.make_mesh(
        axis_names=("data",),
        axis_shapes=(2,),
        devices=[jax.devices()[0], jax.devices()[1]],
    )
    model_mesh = jax.make_mesh(
        axis_names=("model",),
        axis_shapes=(2,),
        devices=[jax.devices()[2], jax.devices()[3]],
    )
    sharding = {
        "data": jsd.NamedSharding(data_mesh, jsd.PartitionSpec("data")),
        "model": jsd.NamedSharding(model_mesh, jsd.PartitionSpec("model")),
    }

    model = eqx.nn.MLP(
        in_size=10,
        out_size=1,
        width_size=5,
        depth=2,
        key=jax.random.PRNGKey(train_seed),
    )

    def test_no_sharding(self):
        pass

    def test_no_validation(self):
        pass

    def test_sharding_from_predefined(self):
        # TODO: initialize data + model sharding and pass
        pass

    def test_sharding_from_devices(self):
        # TODO: pass device list for data + model sharding
        pass

    def test_sharding_single_device(self):
        # TODO: pass single device for data + model sharding
        # => also list with a single device!
        pass
