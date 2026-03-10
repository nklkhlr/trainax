import logging
import shutil
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal, TextIO

import jax
import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray
from tqdm import tqdm

from trainax._file_handler import FileHandler
from trainax._types import EpochOutput, StepOutput, ValStepOutput


def _train_loss_msg(epoch_output: EpochOutput) -> str:
    """Format the training loss for human-readable logging."""
    return f"train loss={epoch_output.train_loss:.4E}"


class Callback:
    """Minimal base class for Trainax callbacks.

    Subclass this and override the hook methods you need. The trainer
    registers callbacks by their :attr:`name` and calls the appropriate hooks
    at each stage of the training loop.

    Attributes
    ----------
    name : str
        Unique identifier used to look up the callback via
        :meth:`~trainax.Trainer.get_callback`.
    val_every : int
        Validation frequency in epochs, kept in sync with the trainer's own
        ``val_every`` setting.

    Methods
    -------
    on_epoch_start(**kwargs)
        Called at the start of each epoch.
    on_epoch_end(model, pbar, epoch, epoch_output, file_handler)
        Called after each epoch with aggregated results.
    on_train_step_end(step_output, **kwargs)
        Called after each training batch step.
    on_val_step_start(step_output, **kwargs)
        Called before each validation batch step.
    on_val_step_end(step_output, **kwargs)
        Called after each validation batch step.
    on_val_start(**kwargs)
        Called before the validation loop begins.
    on_val_end(**kwargs)
        Called after the validation loop completes.
    on_train_end(**kwargs)
        Called once after all epochs finish.
    """

    name: str
    _val_every: int

    def __init__(self, name: str, val_every: int = 1):
        """Assign a unique ``name`` used for registry lookups.

        Parameters
        ----------
        name : str
            Identifier for this callback instance.
        val_every : int, 1
            Validation frequency in epochs. Kept in sync with the trainer.

        Returns
        -------
        None
        """
        self.name = name
        self.val_every = val_every

    @property
    def val_every(self) -> int:
        """Validation frequency in epochs.

        Returns
        -------
        int
            Number of epochs between validation runs.
        """
        return self._val_every

    @val_every.setter
    def val_every(self, val_every: int):
        """Set the validation frequency.

        Parameters
        ----------
        val_every : int
            New validation frequency in epochs.
        """
        self._val_every = val_every

    # TODO: define interfaces for missing methods
    def on_epoch_start(self, *args, **kwargs):
        """Execute callback at the beginning of each epoch."""
        pass

    def on_epoch_end(
        self,
        model: Callable,
        pbar: tqdm,
        epoch: int,
        epoch_output: EpochOutput,
        file_handler: FileHandler,
    ):
        """Execute callback after each epoch with aggregated results.

        Parameters
        ----------
        model : Callable
            Current model (after parameter update).
        pbar : tqdm
            Epoch-level progress bar instance.
        epoch : int
            Zero-based index of the completed epoch.
        epoch_output : EpochOutput
            Aggregated training (and optionally validation) results.
        file_handler : FileHandler
            Open file handler providing write access to any registered output
            files.
        """
        pass

    def on_train_step_end(self, step_output: StepOutput, **kwargs):
        """Execute callback after each training batch step.

        Parameters
        ----------
        step_output : StepOutput
            Output from the completed training step.
        **kwargs
            Additional keyword arguments forwarded by the trainer.
        """
        pass

    def on_val_step_start(self, step_output: StepOutput, **kwargs):
        """Execute callback before each validation batch step.

        Parameters
        ----------
        step_output : StepOutput
            Output from the most recent training step.
        **kwargs
            Additional keyword arguments forwarded by the trainer.
        """
        pass

    def on_val_step_end(self, step_output: StepOutput, **kwargs):
        """Execute callback after each validation batch step.

        Parameters
        ----------
        step_output : StepOutput
            Output from the completed validation step.
        **kwargs
            Additional keyword arguments forwarded by the trainer.
        """
        pass

    def on_val_start(self, *args, **kwargs):
        """Execute callback before validation begins."""
        pass

    def on_val_end(self, *args, **kwargs):
        """Execute callback after validation completes."""
        pass

    def on_train_end(self, *args, **kwargs):
        """Execute callback at the end of training."""
        pass


class EpochLogger(Callback):
    """Log training/validation summaries through to console/file.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance used to emit epoch summaries.
    val_every : int
        Validation frequency inherited from the trainer (see
        :class:`Callback`).

    Methods
    -------
    on_epoch_end(model, pbar, epoch, epoch_output, file_handler)
        Log a formatted epoch summary.
    """

    logger: logging.Logger
    _val_message: Callable[[EpochOutput], str]
    _last_val_loss: float

    def __init__(
        self,
        logger: logging.Logger | None = None,
        name: str = "EpochLogger",
        has_validation: bool = True,
    ):
        """Initialise an epoch-level logger callback.

        Parameters
        ----------
        logger : logging.Logger | None, None
            Logger to write to. A new logger named ``name`` is created when
            ``None``.
        name : str, "EpochLogger"
            Callback name used for registry lookups.
        has_validation : bool, True
            When ``True``, the last known validation loss is appended to each
            log line.

        Returns
        -------
        None
        """
        super().__init__(name)

        if logger is None:
            logger = logging.getLogger(name)
        self.logger = logger

        self._last_val_loss = np.nan
        if has_validation:
            self._val_message = self._val_msg
        else:
            self._val_message = lambda _: ""

    def _val_msg(self, epoch_output: EpochOutput) -> str:
        val_loss = epoch_output.val_loss or self._last_val_loss
        self._last_val_loss = val_loss
        return f", (last) val loss={val_loss:.4E}"

    def on_epoch_end(
        self,
        model: Callable,
        pbar: tqdm,
        epoch: int,
        epoch_output: EpochOutput,
        file_handler: FileHandler,
    ):
        """Log the outcome of an epoch via the configured logger.

        Parameters
        ----------
        model : Callable
            Current model after the parameter update.
        pbar : tqdm
            Epoch-level progress bar (not used directly).
        epoch : int
            Zero-based index of the completed epoch.
        epoch_output : EpochOutput
            Aggregated training and optional validation results.
        file_handler : FileHandler
            Open file handler (not used directly).
        """
        msg = f"Epoch {epoch}: "
        self.logger.info(
            msg
            + _train_loss_msg(epoch_output)
            + self._val_message(epoch_output)
        )


class PbarHandler(Callback):
    """Handle updates on ``tqdm`` progress bars.

    Attributes
    ----------
    val_every : int
        Validation frequency inherited from the trainer (see
        :class:`Callback`).

    Methods
    -------
    on_epoch_end(model, pbar, epoch, epoch_output, file_handler)
        Update the epoch progress bar postfix with the latest losses.
    on_train_end(pbar, **kwargs)
        Advance the progress bar to completion.
    """

    _val_message: Callable[[EpochOutput], str]
    _last_val_loss: float

    def __init__(self, name: str = "PbarHandler", has_validation: bool = True):
        """Initialise a progress-bar update callback.

        Parameters
        ----------
        name : str, "PbarHandler"
            Callback name used for registry lookups.
        has_validation : bool, True
            When ``True``, the last known validation loss is appended to the
            progress bar postfix.

        Returns
        -------
        None
        """
        super().__init__(name)

        self._last_val_loss = np.nan
        if has_validation:
            self._val_message = self._val_msg
        else:
            self._val_message = lambda _: ""

    def _val_msg(self, epoch_output: EpochOutput) -> str:
        val_loss = epoch_output.val_loss or self._last_val_loss
        self._last_val_loss = val_loss
        return f", (last) val loss={val_loss:.4E}"

    def on_epoch_end(
        self,
        model: Callable,
        pbar: tqdm,
        epoch: int,
        epoch_output: EpochOutput,
        file_handler: FileHandler,
    ):
        """Update the progress bar with the latest loss statistics.

        Parameters
        ----------
        model : Callable
            Current model after the parameter update (not used directly).
        pbar : tqdm
            Epoch-level progress bar whose postfix is updated.
        epoch : int
            Zero-based index of the completed epoch (not used directly).
        epoch_output : EpochOutput
            Aggregated training and optional validation results.
        file_handler : FileHandler
            Open file handler (not used directly).
        """
        msg = _train_loss_msg(epoch_output) + self._val_message(epoch_output)
        pbar.set_postfix_str(f"[{msg}]")

    def on_train_end(self, pbar: tqdm, **kwargs):
        """Advance the progress bar to completion.

        Parameters
        ----------
        pbar : tqdm
            Epoch-level progress bar to advance to its total.
        **kwargs
            Additional keyword arguments (ignored).
        """
        pbar.update(pbar.total)


class LossMetricTracker(Callback):
    """Write loss/metric values to disk and mirror them in memory.

    Attributes
    ----------
    losses : dict[str, list[float]]
        In-memory cache of per-epoch loss values keyed by metric name.

    Methods
    -------
    on_epoch_end(model, pbar, epoch, epoch_output, file_handler)
        Append losses to the in-memory cache and flush them to disk.
    """

    losses: dict[str, list[float]]
    _keys: dict[str, str]

    def __init__(
        self,
        train_file_key: str = "train_loss",
        val_file_key: str = "val_loss",
        name: str = "LossMetricTracker",
        **kwargs,
    ):
        """Initialise a loss-tracking callback.

        Parameters
        ----------
        train_file_key : str, "train_loss"
            Key used to retrieve the training-loss file handle from
            :class:`~trainax.FileHandler`.
        val_file_key : str, "val_loss"
            Key used to retrieve the validation-loss file handle.
        name : str, "LossMetricTracker"
            Callback name used for registry lookups.
        **kwargs
            Additional ``file_key`` mappings forwarded to the internal key
            registry for extra metrics.

        Returns
        -------
        None

        Examples
        --------
        >>> tracker = LossMetricTracker(
        ...     train_file_key="train_loss",
        ...     val_file_key="val_loss",
        ... )
        >>> # Pass matching file paths to the trainer
        >>> trainer = EQXTrainer(
        ...     n_epochs=50,
        ...     callbacks=[tracker],
        ...     continuous_files={
        ...         "train_loss": "train.csv",
        ...         "val_loss": "val.csv",
        ...     },
        ... )
        """
        super().__init__(name)
        self.losses = {
            "train_loss": [],
            "val_loss": [],
        }
        self._keys = {
            "train_loss": train_file_key,
            "val_loss": val_file_key,
            **kwargs,
        }

    @staticmethod
    def _write_loss(
        file: TextIO, loss: Float | tuple[Int, Float] | list[Int | Float]
    ):
        """Normalize loss formats and write them to disk."""
        if isinstance(loss, float) or hasattr(loss, "item"):
            file.write(f"{loss}\n")
        elif isinstance(loss, tuple):
            file.write(f"{loss[0]},{loss[1]}\n")
        elif isinstance(loss, list):
            loss_str = ",".join(str(val) for val in loss[1:])
            file.write(f"{loss[0]},{loss_str}\n")
        else:
            raise ValueError(f"Invalid loss type: {type(loss)}")

    def on_epoch_end(
        self,
        model: Callable,
        pbar: tqdm,
        epoch: int,
        epoch_output: EpochOutput,
        file_handler: FileHandler,
    ):
        """Write losses/metrics to disk and cache them in memory.

        Parameters
        ----------
        model : Callable
            Current model (not used directly).
        pbar : tqdm
            Epoch-level progress bar (not used directly).
        epoch : int
            Zero-based index of the completed epoch (not used directly).
        epoch_output : EpochOutput
            Aggregated training and optional validation results.
        file_handler : FileHandler
            Open file handler providing write access to the registered loss
            files.
        """
        self._write_loss(
            file_handler[self._keys["train_loss"]], epoch_output.train_loss
        )
        self.losses[self._keys["train_loss"]].append(epoch_output.train_loss)

        if epoch_output.val_loss is not None:
            self._write_loss(
                file_handler[self._keys["val_loss"]], epoch_output.val_loss
            )
            self.losses[self._keys["train_loss"]].append(epoch_output.val_loss)

        if epoch_output.metrics:
            for key, value in epoch_output.metrics.items():
                self._write_loss(file_handler[key], value)


class BestModelSaver(Callback):
    """Checkpoint the best model based on a monitored metric.

    Attributes
    ----------
    save_model : Callable
        User-supplied function called with ``(model, epoch=epoch)`` whenever
        a new best is found.
    best_value : float
        Best metric value seen so far (property).
    best_epoch : int | None
        Epoch index at which the best model was saved (property).
    key : str
        Metric key currently monitored (property, settable).

    Methods
    -------
    set_key(key)
        Change the metric key used to evaluate model quality.
    on_epoch_end(model, pbar, epoch, epoch_output, file_handler)
        Evaluate the metric and optionally trigger a model save.
    """

    save_model: Callable[..., Any]
    _key: str
    _best_val: float
    _best_iter: int | None
    _get_val: Callable[[EpochOutput, str], float]
    _criterion: Callable[[float, float], bool]

    def __init__(
        self,
        save_fun: Callable[[Callable[..., Any]], None],
        name: str = "BestModelSaver",
        key: Literal["train_loss", "val_loss"] | str = "val_loss",
        criterion: Literal["min", "max"]
        | Callable[[float, float], bool] = "min",
        val_every: int = 1,
    ):
        """Initialize callback to save best model state.

        Parameters
        ----------
        save_fun : Callable
            Function called with ``(model, epoch=epoch)`` whenever a new best
            model is found. Must persist the model to disk.
        name : str, "BestModelSaver"
            Callback name used for registry lookups.
        key : Literal["train_loss", "val_loss"] | str, "val_loss"
            Metric key to monitor. Use ``"train_loss"`` or ``"val_loss"`` for
            built-in losses, or a custom string matching a key in
            ``EpochOutput.metrics``.
        criterion : Literal["min", "max"] | Callable[[float, float], bool], "min"
            Criterion used to determine whether a new metric value is better.
            If a callable is provided it must have the signature
            ``f(new, old) -> bool`` where ``True`` means *new is better*.
        val_every : int, 1
            Save only at epochs that are multiples of this value.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``criterion`` is a string other than ``"min"`` or ``"max"``.

        Examples
        --------
        >>> import pickle, equinox as eqx
        >>> def save_fn(model, epoch):
        ...     with open(f"best_model.pkl", "wb") as f:
        ...         pickle.dump(model, f)
        >>> saver = BestModelSaver(
        ...     save_fun=save_fn,
        ...     key="val_loss",
        ...     criterion="min",
        ... )
        """
        super().__init__(name, val_every=val_every)

        self.save_model = save_fun
        self.key = key

        if isinstance(criterion, str):
            match criterion:
                case "min":
                    self._criterion = self._check_min
                    self._best_val = np.inf
                    self._best_iter = None
                case "max":
                    self._criterion = self._check_max
                    self._best_val = -np.inf
                    self._best_iter = None
                case _:
                    raise ValueError(f"Invalid criterion: {criterion}")
        else:
            self._criterion = criterion

    @staticmethod
    def _check_min(x: float, y: float) -> bool:
        return x < y

    @staticmethod
    def _check_max(x: float, y: float) -> bool:
        return x > y

    @property
    def best_value(self) -> float:
        """Best metric value seen so far.

        Returns
        -------
        float
            The best (minimum or maximum) metric value across all evaluated
            epochs.
        """
        return self._best_val

    @property
    def best_epoch(self) -> int | None:
        """Epoch index at which the best model was saved.

        Returns
        -------
        int | None
            Zero-based epoch index of the last checkpoint, or ``None`` if no
            checkpoint has been saved yet.
        """
        return self._best_iter

    @property
    def key(self):
        """Metric key currently monitored.

        Returns
        -------
        str
            The metric key used to decide when to save the model.
        """
        return self._key

    @key.setter
    def key(self, key: str):
        """Set the metric key used to evaluate model quality.

        Parameters
        ----------
        key : str
            New metric key. Use ``"train_loss"``, ``"val_loss"``, or a custom
            key matching a metric in ``EpochOutput.metrics``.
        """
        self.set_key(key)

    @staticmethod
    def _get_val_loss(epoch_output: EpochOutput, _) -> float:
        return epoch_output.val_loss  # type: ignore[return-value]

    @staticmethod
    def _get_train_loss(epoch_output: EpochOutput, _) -> float:
        return epoch_output.train_loss

    @staticmethod
    def _get_val_metric(epoch_output: EpochOutput, key: str) -> float:
        try:
            return epoch_output.metrics[key]  # type: ignore[return-value]
        except KeyError as ke:
            raise KeyError(
                f"Invalid key: {key}. Metric not found as output "
                "in train/val epoch."
            ) from ke
        except TypeError as te:
            raise ValueError(
                f"Invalid key: {key}. If no specific metrics are "
                "computed during training, use 'train_loss' or "
                "'val_loss'."
            ) from te

    def set_key(self, key: str):
        """Change the metric key used to evaluate model quality.

        Parameters
        ----------
        key : str
            New metric key. Use ``"train_loss"``, ``"val_loss"``, or a custom
            key matching a metric in ``EpochOutput.metrics``.

        Returns
        -------
        None
        """
        match key:
            case "val_loss":
                self._get_val = self._get_val_loss  # type: ignore
            case "train_loss":
                self._get_val = self._get_train_loss
            case _:
                self._get_val = self._get_val_metric
        self._key = key

    def on_epoch_end(
        self,
        model: Callable[..., Any],
        pbar: tqdm,
        epoch: int,
        epoch_output: EpochOutput,
        file_handler: FileHandler,
    ):
        """Evaluate the metric and optionally trigger a model save.

        Parameters
        ----------
        model : Callable
            Current model after the parameter update.
        pbar : tqdm
            Epoch-level progress bar (not used directly).
        epoch : int
            Zero-based index of the completed epoch.
        epoch_output : EpochOutput
            Aggregated training and optional validation results.
        file_handler : FileHandler
            Open file handler (not used directly).

        Raises
        ------
        ValueError
            If the monitored metric value is ``None`` for the current epoch
            (e.g. ``"val_loss"`` is tracked but no validation was run, or
            a custom metric key is not present in ``EpochOutput.metrics``).
        """
        if epoch % self.val_every != 0 and self.key.startswith("val"):
            return

        new_val = self._get_val(epoch_output, self._key)
        try:
            if self._criterion(new_val, self._best_val):
                self._best_val = new_val
                self._best_iter = epoch
                self.save_model(model, epoch=epoch)
        except TypeError as te:
            if new_val is None:
                if self._key == "val_loss":
                    raise ValueError(
                        "No value found for current epoch and key 'val_loss'. "
                        "If no validation loss is computed during training, "
                        "use 'train_loss'."
                    ) from te
                raise ValueError(
                    f"No value found for current epoch and key {self._key}. "
                    "If no specific metrics are "
                    "computed during training, use 'train_loss' or "
                    "'val_loss'."
                ) from te
            raise te


class NNXBestModelSaver(BestModelSaver):
    """Save the best ``flax.nnx`` model state based on a given criterion.

    .. Note::
       This callback requires the ``flax`` and ``orbax`` packages to be
       installed.

    Attributes
    ----------
    save_file : Path
        Path to the directory where the best model checkpoint is written.
    to_cpu : bool
        Whether checkpoints are moved to CPU before saving (property).
    single_device : bool
        Whether checkpoints are gathered onto one device before saving
        (property).
    best_value : float
        Best metric value seen so far (inherited property).
    best_epoch : int | None
        Epoch index of the last checkpoint (inherited property).
    key : str
        Monitored metric key (inherited property).

    Methods
    -------
    load_model(save_file, model_cls, init_params=None, mesh=None, device=None)
        Restore a saved model from an orbax checkpoint directory.
    """

    save_file: Path
    _single_device: bool
    _to_cpu: bool

    def __init__(
        self,
        save_path: str | Path,
        name: str = "NNXBestModelSaver",
        key: Literal["train_loss", "val_loss"] | str = "val_loss",
        criterion: Literal["min", "max"] = "min",
        val_every: int = 1,
        force_overwrite: bool = False,
        save_to_single_device: bool = False,
        save_to_cpu: bool = False,
    ):
        """Initialise a callback to checkpoint the best NNX model.

        Parameters
        ----------
        save_path : str | Path
            Directory in which the ``best_model`` checkpoint subdirectory is
            created.
        name : str, "NNXBestModelSaver"
            Callback name used for registry lookups.
        key : Literal["train_loss", "val_loss"] | str, "val_loss"
            Metric key to monitor.
        criterion : Literal["min", "max"], "min"
            Whether to treat lower (``"min"``) or higher (``"max"``) values as
            better.
        val_every : int, 1
            Save only at epochs that are multiples of this value.
        force_overwrite : bool, False
            When ``True``, an existing checkpoint at ``save_path/best_model``
            is silently removed before training begins.
        save_to_single_device : bool, False
            When ``True``, gather model state onto a single device before
            saving.
        save_to_cpu : bool, False
            When ``True``, move model state to CPU before saving. Implies
            ``save_to_single_device=True``.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If ``flax`` or ``orbax`` is not installed.
        ValueError
            If the checkpoint path already exists and ``force_overwrite`` is
            ``False``.
        """
        if not find_spec("flax"):
            raise ImportError(
                "NNXBestModelSaver requires flax (and orbax) to be installed."
            )
        if not find_spec("orbax"):
            raise ImportError(
                "NNXBestModelSaver requires orbax to be installed"
            )

        self._single_device = save_to_single_device
        self.to_cpu = save_to_cpu
        if self._to_cpu:
            self._to_cpu = self._single_device = True

        super().__init__(self._save_model, name, key, criterion, val_every)
        self.save_file = Path(save_path) / "best_model"

        if self.save_file.exists():
            if not force_overwrite:
                raise ValueError(
                    f"Storage location {str(self.save_file)} already exists. "
                    "Either choose a different location or set "
                    "`force_overwrite=True` to overwrite the existing storage."
                )
            if self.save_file.is_dir():
                shutil.rmtree(self.save_file, ignore_errors=True)
            else:
                self.save_file.unlink()

    @property
    def to_cpu(self):
        """Whether checkpoints are moved to CPU before saving.

        Returns
        -------
        bool
            ``True`` if model state is transferred to CPU prior to each save.
        """
        return self._to_cpu

    @to_cpu.setter
    def to_cpu(self, value: bool):
        """Enable or disable CPU transfer before saving.

        Parameters
        ----------
        value : bool
            When ``True``, also sets :attr:`single_device` to ``True``.
        """
        self._to_cpu = value
        if self._to_cpu:
            self.single_device = True

    @property
    def single_device(self):
        """Whether model state is gathered onto one device before saving.

        Returns
        -------
        bool
            ``True`` if state is concentrated on a single device prior to each
            save.
        """
        return self._single_device

    @single_device.setter
    def single_device(self, value: bool):
        """Enable or disable single-device gathering before saving.

        Parameters
        ----------
        value : bool
            New setting. Cannot be set to ``False`` when :attr:`to_cpu` is
            ``True``.

        Raises
        ------
        ValueError
            If ``value`` is ``False`` and :attr:`to_cpu` is ``True``.
        """
        if not value and self._to_cpu:
            raise ValueError(
                "Cannot set `single_device` to False when `to_cpu` is True."
            )
        self._single_device = value

    def _save_model(self, model, *args, **kwargs):
        import orbax.checkpoint as ocp
        from flax import nnx

        state = nnx.state(model)
        if self._single_device:
            device = jax.devices("cpu")[0] if self._to_cpu else jax.devices()[0]
            state = jax.device_put(state, device)

        state = nnx.to_pure_dict(state)
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(self.save_file, state, force=True)

    @staticmethod
    def load_model(
        save_file: str | Path,
        model_cls,
        init_params: dict[str, Any] | None,
        mesh: jax.sharding.Mesh | None = None,
        device: jax.Device | None = None,
    ):
        """Restore a saved model from an orbax checkpoint directory.

        Parameters
        ----------
        save_file : str | Path
            Path to the orbax checkpoint directory (typically
            ``save_path/best_model``).
        model_cls : type | nnx.Module
            Either the model *class* (used with ``init_params`` to build an
            abstract model) or an already-instantiated ``nnx.Module`` whose
            weights are updated in-place from the checkpoint.
        init_params : dict[str, Any] | None, None
            Keyword arguments passed to ``model_cls(...)`` when constructing
            the abstract model. Ignored when ``model_cls`` is an instance.
        mesh : jax.sharding.Mesh | None, None
            Optional device mesh for distributed loading (currently unused).
        device : jax.Device | None, None
            Optional target device for loading (currently unused).

        Returns
        -------
        nnx.Module
            The restored model with weights loaded from the checkpoint.

        Notes
        -----
        When ``model_cls`` is an ``nnx.Module`` instance the weights are
        merged directly via ``nnx.update``. Otherwise an abstract model is
        constructed with ``nnx.eval_shape`` and weights are injected via
        ``nnx.replace_by_pure_dict``.
        """
        import orbax.checkpoint as ocp
        from flax import nnx

        with ocp.StandardCheckpointer() as checkpointer:
            restored_state = checkpointer.restore(save_file)

        if isinstance(model_cls, nnx.Module):
            return nnx.update(model_cls, restored_state)

        abstract_model = nnx.eval_shape(lambda: model_cls(**init_params or {}))
        graphdef, abstract_state = nnx.split(abstract_model)
        nnx.replace_by_pure_dict(abstract_state, restored_state)
        return nnx.merge(graphdef, abstract_state)


class EQXBestModelSaver(BestModelSaver):
    """Save the best `equinox` model state based on a given criterion.

    .. Note::
       This callback requires the `equinox` package to be installed.

    This callback is a subclass of :py:class:`trainax.callbacks.BestModelSaver`.
    It saves the best model state based on a given criterion. The saved
    model is defined by the `save_file` attribute.

    Parameters
    ----------
    save_path : str | Path
        Path to the directory where the best model will be saved.
    name : str, optional
        The name of the callback, by default "EQXBestModelSaver"
    key : Literal["train_loss", "val_loss"] | str, optional
        The key to use to save the best model, by default "val_loss"
    criterion : Literal["min", "max"] | Callable[[float, float], bool], optional
        Criterion used to determine best model. If a callable is provided,
        the function needs to be of the form
        f(new, old) -> bool[<new is better>]

    Attributes
    ----------
    save_file : Path
        Path to the file where the best model will be saved.

    """

    save_file: Path

    def __init__(
        self,
        save_path: str | Path,
        name: str = "EQXBestModelSaver",
        key: Literal["train_loss", "val_loss"] | str = "val_loss",
        criterion: Literal["min", "max"] = "min",
    ):
        """Initialise a callback to checkpoint the best Equinox model.

        Parameters
        ----------
        save_path : str | Path
            Directory in which the ``best_model`` checkpoint file is created.
        name : str, "EQXBestModelSaver"
            Callback name used for registry lookups.
        key : Literal["train_loss", "val_loss"] | str, "val_loss"
            Metric key to monitor.
        criterion : Literal["min", "max"], "min"
            Whether to treat lower (``"min"``) or higher (``"max"``) values as
            better.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If ``equinox`` or ``orbax`` is not installed.
        """
        if not find_spec("equinox"):
            raise ImportError(
                "EQXBestModelSaver requires equinox (and orbax) to be installed."
            )
        if not find_spec("orbax"):
            raise ImportError(
                "EQXBestModelSaver requires orbax to be installed"
            )

        super().__init__(self._save_model, name, key, criterion)
        self.save_file = Path(save_path) / "best_model"

    @staticmethod
    def _save_model(model, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_model(
        save_file: str | Path, model_cls, init_params: dict[str, Any] | None
    ):
        """Load a saved Equinox model (not yet implemented).

        Parameters
        ----------
        save_file : str | Path
            Path to the checkpoint file.
        model_cls : type
            Model class used to reconstruct the module.
        init_params : dict[str, Any] | None
            Keyword arguments for ``model_cls``.

        Raises
        ------
        NotImplementedError
            Always — this method is not yet implemented.
        """
        raise NotImplementedError


class NNXMetricTracker(Callback):
    """Track ``nnx.metrics.Metric`` values across training and validation.

    Attributes
    ----------
    metrics : nnx.metrics.MultiMetric
        The wrapped metric object used to accumulate per-step values.
    history : dict[str, NDArray]
        Per-metric history arrays shaped ``(n_epochs, 2)`` with columns
        ``[epoch_index, value]``. Keyed as ``"train_<metric>"`` or
        ``"val_<metric>"``.
    tracked_metrics : set[str]
        Set of bare metric names without the ``"train_"``/``"val_"`` prefix
        (property).

    Methods
    -------
    on_train_step_end(step_output, **kwargs)
        Accumulate metrics from a training step.
    on_val_end(epoch, data, **kwargs)
        Flush training metrics and accumulate validation step results.
    on_epoch_end(model, pbar, epoch, epoch_output, file_handler)
        Flush accumulated metrics to history and reset for next epoch.
    on_train_end(pbar, **kwargs)
        Convert history lists to arrays and remove empty entries.
    plot_loss(**kwargs)
        Plot the tracked loss metric.
    plot_metric(metric, **kwargs)
        Plot training and (if available) validation curves for a metric.
    __getitem__(key)
        Retrieve history array(s) by metric key.
    """

    __slots__ = ["metrics", "history", "_mode"]

    def __init__(
        self,
        metrics,
        name: str = "NNXMetricTracker",
    ):
        """Initialise a metric-tracking callback for NNX models.

        Parameters
        ----------
        metrics : nnx.metrics.Metric | nnx.metrics.MultiMetric
            A single metric or a ``MultiMetric`` bundle. A single metric is
            automatically wrapped in a ``MultiMetric`` keyed by its class name.
        name : str, "NNXMetricTracker"
            Callback name used for registry lookups.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If ``flax`` is not installed.
        """
        super().__init__(name)

        try:
            from flax import nnx
        except ImportError as ie:
            raise ImportError(
                "NNXMetricTracker requires flax to be installed."
            ) from ie

        if not isinstance(metrics, nnx.metrics.MultiMetric):
            # this allows us to keep a dict
            metrics = nnx.metrics.MultiMetric(
                **{type(metrics).__name__: metrics}
            )
        self.metrics = metrics
        self._mode = "train"

        self.history = {}
        for metric in self.metrics._metric_names:
            for mode in ["train", "val"]:
                self.history[f"{mode}_{metric}"] = []

    def _reset(self, epoch: int):
        for metric, value in self.metrics.compute().items():
            # print(f"{metric} ({self._mode}): {value.item()}")
            self.history[f"{self._mode}_{metric}"].append([epoch, value.item()])

    def on_train_step_end(self, step_output: StepOutput, **kwargs):
        """Accumulate metrics from a completed training step.

        Parameters
        ----------
        step_output : StepOutput
            Output from the training step containing ``loss``, ``yhat``, and
            ``y``.
        **kwargs
            Additional keyword arguments (ignored).
        """
        # add data from single training step
        self.metrics.update(
            loss=step_output.loss, logits=step_output.yhat, labels=step_output.y
        )

    def on_val_end(self, epoch: int, data: list[ValStepOutput], **kwargs):
        """Flush training metrics and accumulate validation step results.

        Parameters
        ----------
        epoch : int
            Zero-based index of the current epoch.
        data : list[ValStepOutput]
            All validation step outputs for the epoch.
        **kwargs
            Additional keyword arguments (ignored).
        """
        # write metric to train history
        self._reset(epoch)
        # set to validation mode and add data from validation steps
        self._mode = "val"
        for res in data:
            self.metrics.update(loss=res.loss, logits=res.yhat, labels=res.y)

    def on_epoch_end(
        self,
        model: Callable,
        pbar: tqdm,
        epoch: int,
        epoch_output: EpochOutput,
        file_handler: FileHandler,
    ):
        """Flush accumulated metrics to history and reset for the next epoch.

        Parameters
        ----------
        model : Callable
            Current model (not used directly).
        pbar : tqdm
            Epoch-level progress bar (not used directly).
        epoch : int
            Zero-based index of the completed epoch.
        epoch_output : EpochOutput
            Aggregated epoch results (not used directly).
        file_handler : FileHandler
            Open file handler (not used directly).
        """
        # write metric to train history if no validation
        # else write metric to val history
        self._reset(epoch)
        # set to train mode
        self._mode = "train"

    def on_train_end(self, pbar: tqdm, **kwargs):
        """Convert history lists to arrays and remove empty entries.

        Parameters
        ----------
        pbar : tqdm
            Epoch-level progress bar (not used directly).
        **kwargs
            Additional keyword arguments (ignored).
        """
        to_remove = []
        for k, v in self.history.items():
            if not v:
                to_remove.append(k)
            else:
                self.history[k] = np.array(v)
        for k in to_remove:
            del self.history[k]

    @property
    def tracked_metrics(self) -> set[str]:
        """Set of metric names without the ``train_``/``val_`` prefix.

        Returns
        -------
        set[str]
            Bare metric names (e.g. ``{"loss", "accuracy"}``).
        """
        return {k.split("_")[1] for k in self.history}

    def __getitem__(self, key: str) -> NDArray | tuple[NDArray, NDArray]:
        """Retrieve history array(s) by metric key.

        Parameters
        ----------
        key : str
            Either a fully qualified key (``"train_loss"``, ``"val_loss"``) or
            a bare metric name (``"loss"``). A bare name returns both train and
            val arrays as a tuple when both exist, or just the train array.

        Returns
        -------
        NDArray | tuple[NDArray, NDArray]
            Shape ``(n_epochs, 2)`` array with columns ``[epoch, value]``, or
            a ``(train_array, val_array)`` tuple for bare metric names.

        Raises
        ------
        KeyError
            If the requested metric is not tracked.
        """
        try:
            return self.history[key]
        except KeyError as ke:
            if not key.startswith("train_") and not key.startswith("val_"):
                train = self.history[f"train_{key}"]
                try:
                    val = self.history[f"val_{key}"]
                except KeyError:
                    return train
                return train, val

            raise KeyError(
                f"Metric {key.split('_')[1]} is not tracked by NNXMetricTracker"
                f". Tracked metrics: {list(self.tracked_metrics)}"
            ) from ke

    def plot_loss(self, **kwargs):
        """Plot the tracked loss metric.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`plot_metric`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object containing the loss curves.
        """
        return self.plot_metric("loss", **kwargs)

    def plot_metric(self, metric: str, ax=None, **kwargs):
        """Plot training and (if available) validation curves for a metric.

        Parameters
        ----------
        metric : str
            Bare metric name (e.g. ``"loss"``). Must be present in
            :attr:`tracked_metrics`.
        ax : matplotlib.axes.Axes | None, None
            Axes to draw on. A new figure is created when ``None``.
        **kwargs
            Additional keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plotted metric curves.

        Raises
        ------
        ImportError
            If ``matplotlib`` is not installed.
        KeyError
            If ``metric`` is not tracked.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as ie:
            raise ImportError(
                "NNXMetricTracker.plot_metric requires matplotlib to be "
                "installed."
            ) from ie

        ax = ax or plt.figure(figsize=(8, 8)).add_subplot()
        ax.plot(*self[f"train_{metric}"].T, label=f"train {metric}", **kwargs)  # type: ignore

        try:
            ax.plot(*self[f"val_{metric}"].T, label=f"val {metric}", **kwargs)  # type: ignore
            ax.legend()
        except KeyError:
            pass

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())

        return ax
