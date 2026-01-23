import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from jax.tree_util import register_dataclass, tree_map
from jaxtyping import Array, Float, PyTree
from numpy.typing import NDArray

PathLike = os.PathLike | Path | str


def _dataclass_pytrees_to_cpu(inst):
    def _to_numpy_python(arr: Array) -> NDArray | float | int:
        try:
            return arr.item()
        except ValueError:
            return np.array(arr)

    cpu_inst = tree_map(
        _to_numpy_python,
        inst,
        is_leaf=lambda x: isinstance(x, Array),
    )
    for field in fields(inst):
        setattr(inst, field.name, getattr(cpu_inst, field.name))


@dataclass
class StepOutput:
    loss: float | Array
    y: NDArray | Array
    yhat: NDArray | Array
    gradients: PyTree | None = None
    state: PyTree | None = None
    metrics: dict[str, float] | PyTree | None = None

    def to_cpu(self):
        _dataclass_pytrees_to_cpu(self)


@dataclass
class ValStepOutput:
    loss: float | Array
    y: NDArray | Array
    yhat: NDArray | Array
    state: PyTree | None = None
    metrics: dict[str, float] | PyTree | None = None

    def to_cpu(self):
        _dataclass_pytrees_to_cpu(self)


@dataclass
class EpochOutput:
    train_loss: float
    train_losses: NDArray
    y: NDArray
    yhat: NDArray
    gradients: list[PyTree]
    val_loss: float | None = None
    val_y: NDArray | None = None
    val_yhat: NDArray | None = None
    metrics: dict[str, float] | None = None

    @classmethod
    def from_step_outputs(
        cls,
        train_steps: list[StepOutput],
        agg_fun: Callable[[Float[NDArray | Array, " n_batches"]], float],
        val_steps: list[ValStepOutput] | None = None,
    ):
        if not train_steps:
            raise ValueError("At least one training step output is required.")

        train_losses = np.array([np.asarray(step.loss) for step in train_steps])
        train_loss = agg_fun(train_losses)

        if val_steps is not None:
            val_losses = np.array([np.asarray(step.loss) for step in val_steps])
            val_loss = agg_fun(val_losses)
            val_y = np.hstack([np.asarray(step.y) for step in val_steps])
            val_yhat = np.hstack([np.asarray(step.yhat) for step in val_steps])
        else:
            val_loss = None
            val_y = None
            val_yhat = None

        return EpochOutput(
            train_loss=train_loss,
            train_losses=train_losses,
            y=np.hstack([np.asarray(step.y) for step in train_steps]),
            yhat=np.hstack([np.asarray(step.yhat) for step in train_steps]),
            gradients=[step.gradients for step in train_steps],
            val_loss=val_loss,
            val_y=val_y,
            val_yhat=val_yhat,
        )


@dataclass
class TrainOutput:
    model: PyTree
    state: PyTree | None

    def __iter__(self):
        return iter((self.model, self.state))


register_dataclass(StepOutput)
register_dataclass(ValStepOutput)
register_dataclass(EpochOutput)
register_dataclass(TrainOutput)
