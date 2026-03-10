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
    """Output container for a single training step, registered as a JAX pytree.

    Attributes
    ----------
    loss : float | Array
        Scalar loss value for this step.
    y : NDArray | Array
        Ground-truth targets for the batch.
    yhat : NDArray | Array
        Model predictions for the batch.
    gradients : PyTree | None, None
        Raw parameter gradients returned by the step function.
    state : PyTree | None, None
        Updated mutable model state (e.g. BatchNorm statistics), or ``None``
        for stateless models.
    metrics : dict[str, float] | PyTree | None, None
        Optional per-step auxiliary metrics.

    Methods
    -------
    to_cpu()
        Move all array fields to CPU numpy in-place.

    Examples
    --------
    Return a ``StepOutput`` from a training step:

    >>> import jax.numpy as jnp
    >>> loss = jnp.array(0.42)
    >>> y = jnp.ones((8,))
    >>> yhat = jnp.zeros((8,))
    >>> out = StepOutput(loss=loss, y=y, yhat=yhat, gradients=grads)
    """

    loss: float | Array
    y: NDArray | Array
    yhat: NDArray | Array
    gradients: PyTree | None = None
    state: PyTree | None = None
    metrics: dict[str, float] | PyTree | None = None

    def to_cpu(self):
        """Move all array fields to CPU numpy in-place.

        Returns
        -------
        None
        """
        _dataclass_pytrees_to_cpu(self)


@dataclass
class ValStepOutput:
    """Output container for a single validation step, registered as a JAX pytree.

    Attributes
    ----------
    loss : float | Array
        Scalar loss value for this step.
    y : NDArray | Array
        Ground-truth targets for the batch.
    yhat : NDArray | Array
        Model predictions for the batch.
    state : PyTree | None, None
        Updated mutable model state after the forward pass, or ``None`` for
        stateless models.
    metrics : dict[str, float] | PyTree | None, None
        Optional per-step auxiliary metrics.

    Methods
    -------
    to_cpu()
        Move all array fields to CPU numpy in-place.
    """

    loss: float | Array
    y: NDArray | Array
    yhat: NDArray | Array
    state: PyTree | None = None
    metrics: dict[str, float] | PyTree | None = None

    def to_cpu(self):
        """Move all array fields to CPU numpy in-place.

        Returns
        -------
        None
        """
        _dataclass_pytrees_to_cpu(self)


@dataclass
class EpochOutput:
    """Aggregated output over all steps in one training epoch.

    Attributes
    ----------
    train_loss : float
        Aggregated training loss for the epoch (using the configured
        ``aggregate_steps`` strategy).
    train_losses : NDArray
        Per-batch training losses, shape ``(n_batches,)``.
    y : NDArray
        Concatenated ground-truth targets across all training batches.
    yhat : NDArray
        Concatenated model predictions across all training batches.
    gradients : list[PyTree]
        Per-batch gradient pytrees; entries are ``None`` when
        ``keep_gradients=False``.
    val_loss : float | None, None
        Aggregated validation loss, or ``None`` if validation was skipped.
    val_y : NDArray | None, None
        Concatenated validation targets, or ``None`` if validation was skipped.
    val_yhat : NDArray | None, None
        Concatenated validation predictions, or ``None`` if validation was
        skipped.
    metrics : dict[str, float] | None, None
        Optional aggregated auxiliary metrics for the epoch.

    Methods
    -------
    from_step_outputs(train_steps, agg_fun, val_steps=None)
        Construct an instance by aggregating lists of step outputs.
    """

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
        """Construct an instance by aggregating lists of step outputs.

        Parameters
        ----------
        train_steps : list[StepOutput]
            Non-empty list of per-batch training results.
        agg_fun : Callable[[NDArray], float]
            Aggregation function applied to per-batch loss arrays (e.g.
            ``np.nanmean``).
        val_steps : list[ValStepOutput] | None, None
            Per-batch validation results, or ``None`` when validation was
            skipped this epoch.

        Returns
        -------
        EpochOutput
            Aggregated epoch-level summary.

        Raises
        ------
        ValueError
            If ``train_steps`` is empty.
        """
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
    """Two-element container returned by :meth:`~trainax.Trainer.train`.

    Supports unpacking as ``model, state = trainer.train(...)``.

    Attributes
    ----------
    model : PyTree
        The trained model after all epochs.
    state : PyTree | None
        Final mutable model state, or ``None`` for stateless models.

    Methods
    -------
    __iter__()
        Iterate as ``(model, state)`` to support tuple unpacking.
    """

    model: PyTree
    state: PyTree | None

    def __iter__(self):
        """Iterate as ``(model, state)`` to support tuple unpacking.

        Returns
        -------
        Iterator
            Iterator over ``(model, state)``.
        """
        return iter((self.model, self.state))


register_dataclass(StepOutput)
register_dataclass(ValStepOutput)
register_dataclass(EpochOutput)
register_dataclass(TrainOutput)
