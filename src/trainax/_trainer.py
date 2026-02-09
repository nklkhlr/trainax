import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import jax
from jax.experimental import mesh_utils
import jax.sharding as jsd
import numpy as np
from jaxtyping import Array, PyTree
from numpy.typing import NDArray
from optax import GradientTransformation
from tqdm.auto import tqdm
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


class Trainer(ABC):
    """Main training handler.

    Deals with training, validation, and callback handling.

    Attributes
    ----------
    callbacks : dict[str, Callback]
        Mapping of callback names to callback instances.
    file_handler : FileHandler
        File handler for continuously writing and logging via callbacks.
    n_epochs : int
        Number of epochs to iterate through.
    val_every : int
        Frequency (in epochs) for running validation steps.
    use_rich : bool
        Enable rich progress bars when available.
    epoch_state_file: Path | None
        File path to save the trainer state after each epoch. If None, the
        state is not saved during training.
    """

    callbacks: dict[str, Callback]
    file_handler: FileHandler

    n_epochs: int
    val_every: int
    use_rich: bool
    epoch_state_file: Path | None

    _agg_funs: dict[str, Callable] = {
        "mean": np.nanmean,
        "min": np.nanmin,
        "max": np.nanmax,
    }
    _aggregate_steps: Literal["mean", "min", "max"]
    _sharding: dict[str, jsd.NamedSharding | jsd.SingleDeviceSharding | None]
    _jit_fun: Callable
    _val_pbar: tqdm | None

    def __init__(
        self,
        _jit_fun,
        n_epochs: int,
        callbacks: list[Callback],
        continuous_files: dict[str, PathLike] | None = None,
        val_every: int = 5,
        use_rich: bool = True,
        model_sharding: int | list[jax.Device] | None = None,
        data_sharding: int | list[jax.Device] | None = None,
        aggregate_steps: Literal["mean", "min", "max"] = "mean",
        epoch_state_file: str | None = None,
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
        val_every : int, 5
            Frequency (in epochs) for running validation steps.
        use_rich : bool, True
            Enable rich progress bars when available.
        model_sharding : int | list[jax.Device], optional
            Number of devices to shard model across
        data_sharding : int | list[jax.Device], optional
            Number of devices to shard data across
        aggregate_steps : {"mean", "min", "max"}, "mean"
            Aggregation strategy used when reducing batch metrics to epoch
            summaries.
        epoch_state_file: str, optional
            Path to save the trainer state after each epoch. If None, the state
            is not saved during training.
        """
        self._jit_fun = _jit_fun

        self.callbacks = {callback.name: callback for callback in callbacks}

        self.file_handler = FileHandler(continuous_files or {})
        self.n_epochs = n_epochs
        self.val_every = val_every
        self.use_rich = use_rich

        if epoch_state_file is not None:
            self.epoch_state_file = Path(epoch_state_file)
            if not self.epoch_state_file.parent.exists():
                raise FileNotFoundError(
                    f"Directory {self.epoch_state_file.parent} does not exist "
                    "but was provided as parent to `epoch_state_file`."
                )
        else:
            self.epoch_state_file = None

        self._sharding = {}
        self._set_sharding((data_sharding, model_sharding))

        self._aggregate_steps = aggregate_steps
        self._val_pbar = None

    def __getstate__(self):
        if self._sharding:
            mesh_shape = dict(
                zip(
                    self._sharding["data"].mesh.axis_names,
                    self._sharding["data"].mesh.axis_sizes,
                    strict=True,
                )
            )
        else:
            mesh_shape = None

        return {
            "callbacks": self.callbacks,
            "file_handler": self.file_handler,
            "n_epochs": self.n_epochs,
            "val_every": self.val_every,
            "use_rich": self.use_rich,
            "epoch_state_file": self.epoch_state_file,
            "agg_funs": self._agg_funs,
            "aggregate_steps": self._aggregate_steps,
            "mesh_shape": mesh_shape,
            "jit_fun": self._jit_fun,
            "val_pbar": self._val_pbar,
        }

    def __setstate__(self, state):
        self.callbacks = state["callbacks"]
        self.file_handler = state["file_handler"]
        self.n_epochs = state["n_epochs"]
        self.val_every = state["val_every"]
        self.use_rich = state["use_rich"]
        self.epoch_state_file = state["epoch_state_file"]
        self.set_aggregate_steps(state["aggregate_steps"])

        self._sharding = {}
        if (mesh_shape := state["mesh_shape"]) is not None:
            self.sharding = mesh_shape

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
        devices: tuple[int, int] | tuple[list[jax.Device], list[jax.Device]],
    ):
        if isinstance(devices[0], int):
            mesh_shape = devices
            devs = [
                jax.devices()[i]
                for i in range(devices[0] * devices[1])  # type: ignore[invalid-argument]
            ]
        else:
            mesh_shape = (len(devices[0]), len(devices[1]))  # type: ignore[invalid-argument]
            devs = devices[0] + devices[1]  # type: ignore[invalid-argument]

        mesh = jsd.Mesh(
            mesh_utils.create_device_mesh(
                mesh_shape,
                devices=devs,
            ),
            ("data", "model"),
        )
        self._sharding["data"] = jsd.NamedSharding(
            mesh, jsd.PartitionSpec("data")
        )
        self._sharding["model"] = jsd.NamedSharding(
            mesh, jsd.PartitionSpec("model")
        )

    def set_sharding(
        self,
        devices: dict[str, int | list[jax.Device]]
        | tuple[int, int]
        | tuple[list[jax.Device], list[jax.Device]]
        | None,
    ):
        """Set data/model sharding."""
        # TODO: note that if sharding is int or list[int] only single dimension
        if devices is None:
            self._sharding = {}
        # sharding is supported
        if isinstance(devices, tuple):
            self._set_sharding(devices)
        else:
            self._set_sharding((devices["data"], devices["model"]))

    @property
    def sharding(
        self,
    ) -> dict[str, jsd.NamedSharding | jsd.SingleDeviceSharding | None]:
        """dict[str, Sharding | None]: Current data/model sharding settings."""
        return self._sharding

    @sharding.setter
    def sharding(
        self,
        devices: dict[str, int | list[jax.Device]]
        | tuple[int, int]
        | tuple[list[jax.Device], list[jax.Device]]
        | None,
    ):
        self.set_sharding(devices)

    def _epoch_pbar(self, **kwargs) -> tqdm | rich_tqdm:
        tqdm_fun = rich_tqdm if self.use_rich else tqdm
        desc = "Training epochs"
        return tqdm_fun(
            total=self.n_epochs,
            desc=desc,
            bar_format=(
                "{desc} [{n:d}/{total_fmt} | {percentage:3.0f}%] "
                "{bar} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
            **kwargs,
        )

    def _step_pbar(self, loader: JaxLoader, **kwargs) -> tqdm | rich_tqdm:
        tqdm_fun = rich_tqdm if self.use_rich else tqdm
        return tqdm_fun(
            desc=kwargs.pop("desc", "Epoch steps"),
            total=len(loader),
            bar_format=(
                "{desc} [{n:d}/{total_fmt} | {percentage:3.0f}%] "
                "{bar} [{elapsed}<{remaining}, {rate_fmt}]"
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
        **kwargs,
    ) -> list[ValStepOutput]:
        step_results: list[ValStepOutput] = []

        if epoch % self.val_every == 0 and valloader is not None:
            for callback in self.callbacks.values():
                callback.on_val_start(epoch=epoch, loader=valloader)
            pbar = self._val_pbar or self._step_pbar(
                valloader, desc="Validation steps"
            )
            pbar.reset(total=len(valloader))
            for data in valloader:
                output = val_step(model, data, **kwargs)
                output.to_cpu()
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

    def _jit_val_step(
        self,
        valloader: JaxLoader | None,
        val_step: StateStepFun | StepFun | None,
        model=None,
    ) -> StateStepFun | StepFun | None:
        if valloader is not None:
            if val_step is not None:
                return self._jit_fun(val_step, model)
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
            "epoch_start",
            "epoch_end",
            "val_step_start",
            "val_step_end",
            "train_step_start",
            "train_step_end",
            "train_end",
        ],
        **kwargs,
    ):
        for callback in self.callbacks.values():
            getattr(callback, "on_" + event)(**kwargs)

    @staticmethod
    @abstractmethod
    def _optim_init(
        optim,
        model: Callable[..., Any],
    ) -> PyTree:
        pass

    @staticmethod
    @abstractmethod
    def _setup_step_fun(
        train_step: StateStepFun | StepFun,
        optim: GradientTransformation,
        model_sharding: jsd.NamedSharding | None = None,
        **kwargs,
    ) -> Callable[..., Any]:
        pass

    @staticmethod
    @abstractmethod
    def _inference_mode(
        model: Callable[..., Any], state: PyTree | None
    ) -> Callable[..., Any]:
        """Wrap model in inference mode."""
        pass

    @staticmethod
    @abstractmethod
    def _train_mode(model: Callable[..., Any], **kwargs) -> Callable[..., Any]:
        """Set model in training mode."""
        pass

    def save(self, file_path: str | os.PathLike | Path) -> None:
        """
        Save the trainer instance to a given directory.

        Parameters
        ----------
        file_path: str
            File to save the trainer instance.

        Returns
        -------
        None
        """
        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    def get_callback(self, name: str) -> Callback:
        """
        Retrieve a callback instance by name.

        Parameters
        ----------
        name : str
            Name of the callback to retrieve.

        Returns
        -------
        Callback
            The callback instance.

        Raises
        ------
        KeyError
            If the requested callback name unknown. Available
            names can be printed with `callbacks.keys()`
        """
        try:
            return self.callbacks[name]
        except KeyError as ke:
            raise KeyError(
                f"Unknown callback {name}. Available options are "
                f"{list(self.callbacks.keys())}"
            ) from ke

    def train(
        self,
        model: Callable[..., Any],
        optim: GradientTransformation,
        train_step: StateStepFun | StepFun,
        trainloader: JaxLoader,
        val_step: StepFun | None,
        valloader: JaxLoader | None,
        train_state: PyTree | None = None,
        keep_gradients: bool = False,
        **kwargs,
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
        keep_gradients: bool, False
            Whether to retain gradients beyond the batch step.

        Returns
        -------
        TrainOutput
            The updated model and final training state (if present).
        """
        agg_fun = self._agg_funs[self._aggregate_steps]
        if valloader is not None:
            if val_step is None:
                raise ValueError(
                    "'valloader' provided but val_step is not defined. "
                    "Please set the validation step function "
                    "(Trainer.val_step)."
                )
            for callback in self.callbacks.values():
                callback.val_every = self.val_every

        self._val_pbar = None

        self.file_handler.open()

        trainloader, valloader = self._prep_data(trainloader, valloader)

        model_sharding = (
            self._sharding["model"] if self._sharding is not None else None
        )
        step_fun, model = self._jit_fun(
            self._setup_step_fun(train_step, optim, model_sharding, **kwargs),
            model,
        )
        opt_state = self._optim_init(optim, model)
        state = train_state

        val_step_fun, model = self._jit_val_step(valloader, val_step, model)  # type: ignore
        epoch_bar = self._epoch_pbar()
        step_bar = self._step_pbar(trainloader)
        for epoch in range(self.n_epochs):
            self._invoke_callbacks(
                event="epoch_start",
                epoch=epoch,
                pbar=epoch_bar,
                file_handler=self.file_handler,
            )

            epoch_data: list[StepOutput] = []
            step_bar.reset()
            for data in trainloader:
                model, output, opt_state, state = step_fun(
                    model, data, opt_state, state
                )
                output.to_cpu()
                del data

                self._invoke_callbacks(
                    event="train_step_end",
                    pbar=step_bar,
                    step_output=output,
                )

                if not keep_gradients:
                    output.gradients = None
                epoch_data.append(output)

                step_bar.update(1)

            step_bar.refresh()

            val_outputs: list[ValStepOutput] | None = None
            if val_step_fun is not None:
                inference_model = self._inference_mode(model, state)
                val_outputs = self._validation(
                    epoch=epoch,
                    model=inference_model,
                    val_step=val_step_fun,
                    valloader=valloader,  # type: ignore
                )
                if not val_outputs:
                    val_outputs = None

            model = self._train_mode(model)
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

            if self.epoch_state_file is not None:
                self.save(self.epoch_state_file)

            epoch_bar.update(1)

        epoch_bar.refresh()

        self.file_handler.close()

        self._invoke_callbacks(
            event="train_end",
            pbar=epoch_bar,
        )

        return TrainOutput(model, state)


class EQXTrainer(Trainer):
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
        jit_kwargs: dict[str, Any] | None = None,
        epoch_state_file: str | None = None,
    ):
        """Initialise a trainer instance for an equinox model.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to iterate through.
        callbacks : list[Callback]
            Callback instances for logging, checkpointing, etc.
        continuous_files : dict[str, PathLike], optional
            Mapping of callback keys to open file paths managed by
            :class:`~trainax._file_handler.FileHandler`.
        val_every : int, 5
            Frequency (in epochs) for running validation steps.
        use_rich : bool, True
            Enable rich progress bars when available.
        model_sharding : list[int] | int | jsd.NamedSharding | None, optional
            Placeholder for future model sharding support. Any non-``None``
            value currently raises an Exception.
        data_sharding : list[int] | int | jsd.NamedSharding | None, optional
            Sharding applied to provided data loaders.
        aggregate_steps : {"mean", "min", "max"}, "mean"
            Aggregation strategy used when reducing batch metrics to epoch
            summaries.
        jit_kwargs: dict[str, Any] | None, None
            Optional keyword arguments for `eqx.filter_jit`
        epoch_state_file: str, optional
            File to save the trainer state after each epoch. If None, the state
            is not saved during training.
        """
        try:
            import equinox as eqx
        except ImportError as ie:
            raise ImportError(
                "Equinox must be installed to use EQXTrainer. "
                "Install with either of "
                "`pip install <trainax[eqx]|trainax[all]|equinox`."
            ) from ie

        super().__init__(
            lambda f, m: (eqx.filter_jit(f, **jit_kwargs or {}), m),
            n_epochs=n_epochs,
            callbacks=callbacks,
            continuous_files=continuous_files,
            val_every=val_every,
            use_rich=use_rich,
            model_sharding=model_sharding,
            data_sharding=data_sharding,
            aggregate_steps=aggregate_steps,
            epoch_state_file=epoch_state_file,
        )

    @staticmethod
    def _optim_init(
        optim: GradientTransformation,
        model: Callable[..., Any],
    ) -> PyTree:
        import equinox as eqx

        return optim.init(eqx.filter(model, eqx.is_array))

    @staticmethod
    def _setup_step_fun(
        train_step: StateStepFun | StepFun,
        optim: GradientTransformation,
        model_sharding: jsd.NamedSharding | None = None,
        **kwargs,
    ) -> Callable[..., Any]:
        import equinox as eqx

        def _fun(
            model_: Callable[..., Any],
            data_: dict[str, NDArray | Array],
            opt_state_: PyTree,
            state_: PyTree | None,
        ) -> tuple[Callable[..., Any], StepOutput, PyTree, PyTree | None]:
            if model_sharding is not None:
                model_, opt_state_, state_ = eqx.filter_shard(
                    (model_, opt_state_, state_), model_sharding
                )

            if state_ is None:
                out = train_step(model_, data_, **kwargs)  # type: ignore
            else:
                out = train_step(model_, data_, state_, **kwargs)  # type: ignore

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

        return _fun

    @staticmethod
    def _inference_mode(
        model: Callable[..., Any], state: PyTree | None
    ) -> Callable[..., Any]:
        import equinox as eqx

        inference_model = eqx.nn.inference_mode(model)
        if state is not None:
            inference_model = eqx.Partial(inference_model, state=state)
        return inference_model

    @staticmethod
    def _train_mode(model: Callable[..., Any], **kwargs) -> Callable[..., Any]:
        import equinox as eqx

        return eqx.nn.inference_mode(model, value=False)


class NNXTrainer(Trainer):
    _jit_kwargs: dict[str, Any]

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
        jit_kwargs: dict[str, Any] | None = None,
        epoch_state_file: str | None = None,
    ):
        """Initialise a trainer instance for a flax.nnx model.

        Parameters
        ----------
        n_epochs : int
            Number of epochs to iterate through.
        callbacks : list[Callback]
            Callback instances for logging, checkpointing, etc.
        continuous_files : dict[str, PathLike], optional
            Mapping of callback keys to open file paths managed by
            :class:`~trainax._file_handler.FileHandler`.
        val_every : int, 5
            Frequency (in epochs) for running validation steps.
        use_rich : bool, True
            Enable rich progress bars when available.
        model_sharding : list[int] | int | jsd.NamedSharding | None, optional
            Placeholder for future model sharding support. Any non-``None``
            value currently raises an Exception.
        data_sharding : list[int] | int | jsd.NamedSharding | None, optional
            Sharding applied to provided data loaders.
        aggregate_steps : {"mean", "min", "max"}, "mean"
            Aggregation strategy used when reducing batch metrics to epoch
            summaries.
        jit_kwargs: dict[str, Any] | None, None
            Optional keyword arguments for `nnx.filter_jit`
        epoch_state_file: str, False
            File to save the trainer state after each epoch. If None, the state
            is not saved during training.
        """
        if find_spec("flax") is None:
            raise ImportError(
                "Flax must be installed to use NNXTrainer. "
                "Install with either of "
                "`pip install <trainax[flax]|trainax[all]`>."
            )

        # TODO: how to put memory donation back in?
        # _default_jit_kwargs = {"donate_argnames": ("model_", "opt_state_")}
        self._jit_kwargs = jit_kwargs or {}

        if data_sharding is None:
            jit_fun = self._jit_no_sharding
        else:
            jit_fun = self._jit_sharding

        super().__init__(
            jit_fun,
            n_epochs=n_epochs,
            callbacks=callbacks,
            continuous_files=continuous_files,
            val_every=val_every,
            use_rich=use_rich,
            model_sharding=model_sharding,
            data_sharding=data_sharding,
            aggregate_steps=aggregate_steps,
            epoch_state_file=epoch_state_file,
        )

    def __getstate__(self):
        state = super().__getstate__()
        state["jit_kwargs"] = self._jit_kwargs
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._jit_kwargs = state["jit_kwargs"]

    def _jit_no_sharding(self, fun, model):
        from flax import nnx

        return nnx.jit(fun, **self._jit_kwargs), model

    def _jit_sharding(self, fun, model):
        from flax import nnx

        mesh = self._sharding["data"].mesh

        @nnx.jit
        def jit_shard(module):
            state = nnx.state(module)
            pspecs = nnx.get_partition_spec(state)
            with mesh:
                sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(module, sharded_state)
            return module

        return nnx.jit(
            fun,
            **self._jit_kwargs,
        ), jit_shard(model)

    def _set_sharding(
        self,
        devices: dict[str, int | list[jax.Device]]
        | tuple[int, int]
        | tuple[list[jax.Device], list[jax.Device]]
        | None,
    ):
        super()._set_sharding(devices)
        if self._sharding:
            self._jit_fun = self._jit_sharding
        else:
            self._jit_fun = self._jit_no_sharding

    @staticmethod
    def _optim_init(
        optim: GradientTransformation,
        model: Callable[..., Any],
    ) -> PyTree:
        from flax import nnx

        return nnx.ModelAndOptimizer(model, optim)  # type: ignore

    @staticmethod
    def _setup_step_fun(
        train_step: StateStepFun | StepFun,
        optim,
        model_sharding: jsd.NamedSharding | None = None,
        **kwargs,
    ) -> Callable[..., Any]:
        from flax import nnx

        def _fun(
            model_: nnx.Module,
            data_: dict[str, NDArray | Array],
            opt_state_: PyTree,
            _: None,
        ) -> tuple[Callable[..., Any], StepOutput, PyTree, PyTree | None]:
            out = train_step(model_, data_, **kwargs)  # type: ignore

            if out.gradients is None:
                raise ValueError(
                    "train_step must return gradients to apply optimizer "
                    "updates."
                )

            opt_state_.update(out.gradients)

            return model_, out, opt_state_, out.state  # type: ignore

        return _fun

    @staticmethod
    def _inference_mode(
        model: Callable[..., Any], state: None
    ) -> Callable[..., Any]:
        model.eval()  # type: ignore
        return model

    @staticmethod
    def _train_mode(model: Callable[..., Any], **kwargs) -> Callable[..., Any]:
        model.train()  # type: ignore
        return model
