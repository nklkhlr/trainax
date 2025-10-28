from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.sharding as jsd
import numpy as np
from jaxtyping import Array, PyTree
from numpy.typing import NDArray
from optax import GradientTransformation
from tqdm import tqdm
from tqdm.rich import tqdm as rich_tqdm

from trainax._callbacks import Callback
from trainax._dataloader import JaxLoader
from trainax._file_handler import FileHandler
from trainax._types import (
    EpochOutput,
    PathLike,
    StepOutput,
    TrainOutput,
    ValStepOutput,
)

StepFun = Callable[
    [Callable[..., Any], dict[str, NDArray | Array]],
    StepOutput,
]
StateStepFun = Callable[
    [Callable[..., Any], dict[str, NDArray | Array], PyTree],
    StepOutput,
]


class Trainer:
    """Main training handler.

    Deals with training, validation, and callback handling.

    Attributes
    ----------
    callbacks : dict[str, Callback]
        Mapping of callback names to callback instances.
    file_handlers : dict[str, FileHandler]
        Mapping of callback names to file handlers.
    n_epochs : int
        Number of epochs to iterate through.
    val_every : int
        Frequency (in epochs) for running validation steps.
    use_rich : bool
        Enable rich progress bars when available.
    """

    callbacks: dict[str, Callback]
    file_handlers: dict[str, FileHandler]

    n_epochs: int
    val_every: int
    use_rich: bool

    _agg_funs: dict[str, Callable] = {
        "mean": np.nanmean,
        "min": np.nanmin,
        "max": np.nanmax,
    }
    _aggregate_steps: Literal["mean", "min", "max"]
    _sharding: dict[str, jsd.NamedSharding | jsd.SingleDeviceSharding | None]

    def __init__(
        self,
        n_epochs: int,
        callbacks: list[Callback],
        continuous_files: dict[str, PathLike] | None = None,
        val_every: int = 5,
        use_rich: bool = True,
        model_sharding: list[int] | int | jsd.NamedSharding | None = None,
        data_sharding: list[int] | int | jsd.NamedSharding | None = None,
        aggregate_steps: Literal["mean", "min", "max"] = "mean",
    ):
        """Initialise a trainer instance.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to iterate through.
        callbacks : list[Callback]
            Callback instances for logging, checkpointing, etc.
        continuous_files : dict[str, PathLike], optional
            Mapping of callback keys to open file paths managed by
            :class:`~trainax._file_handler.FileHandler`.
        val_every : int, default=5
            Frequency (in epochs) for running validation steps.
        use_rich : bool, default=True
            Enable rich progress bars when available.
        model_sharding : list[int] | int | jsd.NamedSharding | None, optional
            Placeholder for future model sharding support. Any non-``None``
            value currently raises an Exception.
        data_sharding : list[int] | int | jsd.NamedSharding | None, optional
            Sharding applied to provided data loaders.
        aggregate_steps : {"mean", "min", "max"}, default="mean"
            Aggregation strategy used when reducing batch metrics to epoch
            summaries.
        """
        self.callbacks = {callback.name: callback for callback in callbacks}

        self.file_handler = FileHandler(continuous_files or {})
        self.n_epochs = n_epochs
        self.val_every = val_every
        self.use_rich = use_rich

        self._sharding = {}
        self._set_sharding(model_sharding, "model")
        self._set_sharding(data_sharding, "data")

        self._aggregate_steps = aggregate_steps

    @property
    def aggregate_steps(self):
        """str: Aggregation strategy applied to per-step metrics."""
        return self._aggregate_steps

    def set_aggregate_steps(
        self, aggregate_steps: Literal["mean", "min", "max"]
    ):
        """Override the aggregation strategy used in epoch summaries."""
        if aggregate_steps not in self._agg_funs:
            raise ValueError(
                f"Invalid aggregate_steps: {aggregate_steps}. "
                f"Must be one of {list(self._agg_funs.keys())}"
            )
        self._aggregate_steps = aggregate_steps

    def _set_sharding(
        self,
        sharding: list[int] | int | jsd.NamedSharding | None,
        kind: Literal["data", "model"],
    ):
        if kind not in ["data", "model"]:
            raise ValueError(
                f"Invalid sharding kind: {kind}. Must be 'data' or 'model'"
            )
        if kind == "model" and sharding is not None:
            raise NotImplementedError("Model sharding is not yet supported")

        if sharding is None:
            self._sharding[kind] = None
        elif isinstance(sharding, jsd.NamedSharding):
            self._sharding[kind] = sharding
        elif isinstance(sharding, list) and len(sharding) > 1:
            devices = [jax.devices()[i] for i in sharding]
            mesh = jax.make_mesh(
                axis_shapes=(len(devices),),
                axis_names=(kind,),
                devices=devices,
            )
            self._sharding[kind] = jsd.NamedSharding(
                mesh, jsd.PartitionSpec(kind)
            )
        else:
            if isinstance(sharding, list):
                sharding = sharding[0]
            self._sharding[kind] = jsd.SingleDeviceSharding(
                jax.devices()[sharding]
            )

    @property
    def sharding(
        self,
    ) -> dict[str, jsd.NamedSharding | jsd.SingleDeviceSharding | None]:
        """dict[str, Sharding | None]: Current data/model sharding settings."""
        return self._sharding

    def set_sharding(self, sharding, kind: Literal["data", "model"]):
        """Set data/model sharding."""
        # TODO: note that if sharding is int or list[int] only single dimension
        # sharding is supported
        self._set_sharding(sharding, kind)

    def _epoch_pbar(self, **kwargs) -> tqdm | rich_tqdm:
        tqdm_fun = rich_tqdm if self.use_rich else tqdm
        desc = "Training epochs"
        return tqdm_fun(
            total=self.n_epochs,
            desc=desc,
            bar_format=(
                "{desc} [{n:d}/{total_fmt} ({percentage:3.0f}%)] | "
                "{bar} [{elapsed}<{remaining}, {rate_fmt}] | {postfix}"
            ),
            **kwargs,
        )

    def _step_pbar(self, loader: JaxLoader, **kwargs) -> tqdm | rich_tqdm:
        tqdm_fun = rich_tqdm if self.use_rich else tqdm
        return tqdm_fun(
            desc=kwargs.pop("desc", "Epoch steps"),
            total=len(loader),
            bar_format=(
                "{desc} [{n:d}/{total_fmt} | {percentage:3.0f}%] | "
                "{bar} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
            leave=kwargs.pop("leave", False),
            **kwargs,
        )

    def _validation(
        self,
        epoch: int,
        model: Callable[..., Any],
        val_step: StepFun,
        valloader: JaxLoader,
    ) -> list[ValStepOutput]:
        step_results: list[ValStepOutput] = []

        for callback in self.callbacks.values():
            callback.on_val_start(epoch=epoch, loader=valloader)

        if (epoch + 1) % self.val_every == 0 and valloader is not None:
            pbar = self._step_pbar(
                valloader, desc="Validation steps", leave=False
            )
            for data in valloader:
                output = val_step(model, data)
                step_results.append(output)  # type: ignore[arg-type]
                pbar.update(1)
            pbar.refresh()

        for callback in self.callbacks.values():
            callback.on_val_end(epoch=epoch, data=step_results)

        return step_results

    def _prep_data(
        self, trainloader: JaxLoader, valloader: JaxLoader | None
    ) -> tuple[JaxLoader, JaxLoader | None]:
        if (data_sharding := self._sharding.get("data")) is not None:
            trainloader.set_sharding(data_sharding)
            if valloader is not None:
                valloader.set_sharding(data_sharding)

        # TODO: add in model sharding on .model

        return trainloader, valloader

    @staticmethod
    def _jit_val_step(
        jit_fun: Callable[
            [Callable[..., Any]],
            Callable[..., Any],
        ],
        valloader: JaxLoader | None,
        val_step: StateStepFun | StepFun | None,
    ) -> StateStepFun | StepFun | None:
        if valloader is not None:
            if val_step is not None:
                return jit_fun(val_step)
            raise ValueError(
                "Validation loader provided but val_step is not defined. "
                "Please set the validation step function "
                "(Trainer.val_step)."
            )
        if val_step is not None:
            raise ValueError(
                "Validation step provided but valloader is not defined. "
                "Please pass a validation loader."
            )

        return None

    def _invoke_callbacks(
        self,
        event: Literal[
            "epoch_start", "epoch_end", "step_start", "step_end", "train_end"
        ],
        **kwargs,
    ):
        for callback in self.callbacks.values():
            getattr(callback, "on_" + event)(**kwargs)

    def train(
        self,
        model: Callable[..., Any],
        optim: GradientTransformation,
        train_step: StateStepFun | StepFun,
        trainloader: JaxLoader,
        val_step: StepFun | None,
        valloader: JaxLoader | None,
        train_state: PyTree | None = None,
        jit_fun: Callable[
            [Callable[..., Any]],
            Callable[..., Any],
        ] = eqx.filter_jit,
    ) -> TrainOutput:
        """Execute the training loop and return the updated model/state.

        Parameters
        ----------
        model : Callable[..., Any]
            Trainable function or module.
        optim : optax.GradientTransformation
            Optimiser providing ``init`` and ``update``.
        train_step : StepFun | StateStepFun
            Callable that computes loss, gradients, and optional state updates.
            Signature should match whether the function is stateful or not.
        trainloader : JaxLoader
            Loader iterating over training batches.
        val_step : StepFun | None
            Optional validation function invoked according to ``val_every``.
            Signature should match whether the function is stateful or not.
        valloader : JaxLoader | None
            Validation loader yielding batches; ignored when ``None``.
        train_state : PyTree | None, optional
            Initial mutable state (e.g. BatchNorm statistics) for training.
        jit_fun : Callable, optional
            Transformation used to JIT the update/validation functions.

        Returns
        -------
        TrainOutput
            The updated model and final training state (if present).
        """
        agg_fun = self._agg_funs[self._aggregate_steps]
        if valloader is not None and val_step is None:
            raise ValueError(
                "'valloader' provided but val_step is not defined. "
                "Please set the validation step function "
                "(Trainer.val_step)."
            )

        trainloader, valloader = self._prep_data(trainloader, valloader)

        def _step(
            model_: Callable[..., Any],
            data_: dict[str, NDArray | Array],
            opt_state_: PyTree,
            state_: PyTree | None,
        ):
            if state_ is None:
                out = train_step(model_, data_)  # type: ignore
            else:
                out = train_step(model_, data_, state_)  # type: ignore

            if out.gradients is None:
                raise ValueError(
                    "train_step must return gradients to apply optimizer "
                    "updates."
                )
            updates, opt_state_ = optim.update(
                out.gradients, opt_state_, eqx.filter(model_, eqx.is_array)
            )
            model_ = eqx.apply_updates(model_, updates)

            return model_, out, opt_state_, out.state

        step_fun = jit_fun(_step)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        state = train_state

        val_step_fun = self._jit_val_step(jit_fun, valloader, val_step)
        epoch_bar = self._epoch_pbar()
        for epoch in range(self.n_epochs):
            self._invoke_callbacks(
                event="epoch_start",
                epoch=epoch,
                pbar=epoch_bar,
                file_handler=self.file_handler,
            )

            epoch_data: list[StepOutput] = []
            step_bar = self._step_pbar(trainloader)
            for data in trainloader:
                model, output, opt_state, state = step_fun(
                    model, data, opt_state, state
                )
                epoch_data.append(output)

                self._invoke_callbacks(
                    event="step_end",
                    pbar=step_bar,
                    step_output=output,
                )
                step_bar.update(1)
            step_bar.refresh()

            val_outputs: list[ValStepOutput] | None = None
            if val_step_fun is not None:
                inference_model = eqx.nn.inference_mode(model)
                if state is not None:
                    inference_model = eqx.Partial(
                        inference_model, state=train_state
                    )

                val_outputs = self._validation(
                    epoch=epoch,
                    model=inference_model,
                    val_step=val_step_fun,
                    valloader=valloader,  # type: ignore
                )
                if not val_outputs:
                    val_outputs = None

            epoch_output = EpochOutput.from_step_outputs(
                epoch_data,
                agg_fun,
                val_outputs,
            )
            self._invoke_callbacks(
                event="epoch_end",
                model=model,
                epoch=epoch,
                pbar=epoch_bar,
                epoch_output=epoch_output,
                file_handler=self.file_handler,
            )
            epoch_bar.update(1)
        epoch_bar.refresh()

        self._invoke_callbacks(
            event="train_end",
            pbar=epoch_bar,
        )

        return TrainOutput(model, state)
