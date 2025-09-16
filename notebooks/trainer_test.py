import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from optax import adam

from trainax import (
    EpochLogger,
    JaxLoader,
    PbarHandler,
    StepOutput,
    Trainer,
    ValStepOutput,
)

jax.config.update("jax_num_cpu_devices", 4)

SEED = 42
NTRAIN = 1000
NVAL = 200
M = 20
BATCH_SIZE = 200
rng = np.random.RandomState(SEED)


def loss_fun(yhat, y):
    return jnp.mean((yhat - y) ** 2)


def make_step(model: eqx.Module, data: dict) -> StepOutput:
    print("make step")
    yhat = jax.vmap(model)(data["x"])
    jax.debug.print("{yshape}", yshape=yhat.shape)
    loss, grads = eqx.filter_value_and_grad(loss_fun)(yhat, data["y"])

    return StepOutput(loss=loss, y=data["y"], yhat=yhat, gradients=grads)


def val_step(model: eqx.Module, data: dict) -> ValStepOutput:
    inference_model = eqx.nn.inference_mode(model)
    yhat = jax.vmap(inference_model)(data["x"])
    loss = loss_fun(yhat, data["y"])
    return ValStepOutput(loss=loss, y=data["y"], yhat=yhat)


train_data = {"x": rng.randn(NTRAIN, M), "y": rng.randint(0, 2, NTRAIN)}
val_data = {"x": rng.randn(NVAL, M), "y": rng.randint(0, 2, NVAL)}

train_loader = JaxLoader(train_data, batch_size=BATCH_SIZE)
val_loader = JaxLoader(val_data, batch_size=BATCH_SIZE)

model = eqx.nn.MLP(
    in_size=M, out_size=1, width_size=10, depth=5, key=jr.PRNGKey(SEED)
)


# TODO: add additional callbacks
callbacks = [
    EpochLogger(),
    PbarHandler(),
]

# TODO: test manual sharding
data_sharding = [0, 1, 2, 3]

# TODO: implement trainer
trainer = Trainer(
    n_epochs=10,
    val_every=2,
    callbacks=callbacks,
    data_sharding=data_sharding,
)

print("=====\nRunning trainer\n=====")
trainer.train(
    model=model,
    optim=adam(1e-3),
    trainloader=train_loader,
    valloader=val_loader,
    train_step=make_step,
    val_step=val_step,
    jit_fun=eqx.filter_jit,
)
