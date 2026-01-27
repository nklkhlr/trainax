import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TextIO

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
    """Minimal base class for Trainax callbacks."""

    name: str
    _val_every: int

    def __init__(self, name: str, val_every: int = 1):
        """Assign a unique `name` used for registry lookups."""
        self.name = name
        self.val_every = val_every

    @property
    def val_every(self) -> int:
        """Return the validation frequency."""
        return self._val_every

    @val_every.setter
    def val_every(self, val_every: int):
        """Set the validation frequency."""
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
        """Execute callback after each epoch with aggregated results."""
        pass

    def on_train_step_end(self, step_output: StepOutput, **kwargs):
        """Execute callback after each training step."""
        pass

    def on_val_step_start(self, step_output: StepOutput, **kwargs):
        """Execute callback after each training step."""
        pass

    def on_val_step_end(self, step_output: StepOutput, **kwargs):
        """Execute callback after each training step."""
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
    """Log training/validation summaries through to console/file."""

    logger: logging.Logger
    _val_message: Callable[[EpochOutput], str]
    _last_val_loss: float

    def __init__(
        self,
        logger: logging.Logger | None = None,
        name: str = "EpochLogger",
        has_validation: bool = True,
    ):
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
        """Log the outcome of an epoch via the configured logger."""
        msg = f"Epoch {epoch}: "
        self.logger.info(
            msg
            + _train_loss_msg(epoch_output)
            + self._val_message(epoch_output)
        )


class PbarHandler(Callback):
    """Handle updates on `tqdm` progress bars."""

    _val_message: Callable[[EpochOutput], str]
    _last_val_loss: float

    def __init__(self, name: str = "PbarHandler", has_validation: bool = True):
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
        """Update the progress bar with the latest loss statistics."""
        msg = _train_loss_msg(epoch_output) + self._val_message(epoch_output)
        pbar.set_postfix_str(f"[{msg}]")

    def on_train_end(self, pbar: tqdm, **kwargs):
        """Ensure progress bar is 'full'."""
        pbar.update(pbar.total)


class LossMetricTracker(Callback):
    """Write loss/metric values to disk and mirror them in memory."""

    losses: dict[str, list[float]]
    _keys: dict[str, str]

    def __init__(
        self,
        train_file_key: str = "train_loss",
        val_file_key: str = "val_loss",
        name: str = "LossMetricTracker",
        **kwargs,
    ):
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
        """Write losses/metrics to disk and cache them in memory."""
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
    """Checkpoint best model."""

    save_model: Callable[..., Any]
    val_every: int
    _key: str
    _best_val: float
    _best_iter: int | None
    _get_val: Callable[[EpochOutput], float]
    _criterion: Callable[[float, float], bool]

    def __init__(
        self,
        save_fun: Callable[[Callable[..., Any]], None],
        name: str = "BestModelSaver",
        key: Literal["train_loss", "val_loss"] | str = "val_loss",
        criterion: Literal["min", "max"] = "min",
        val_every: int = 1,
    ):
        """Initialize callback to save best model state.

        Parameters
        ----------
        name : str, optional
            The name of the callback, by default "BestModelSaver"
        key : Literal["train_loss", "val_loss"] | str, "val_loss"
            The key to use to save the best model, by default "val_loss"
        criterion : Literal["min", "max"] | Callable[[float, float], bool], min
            Criterion used to determine best model. If a callable is provided,
            the function needs to be of the form
            f(new, old) -> bool[<new is better>]
        val_every : int, optional
            How often to evaluate the model, by default 1
        """
        super().__init__(name)

        self.save_model = save_fun
        self.key = key
        self.val_every = val_every

        match criterion:
            case "min":
                self._criterion = lambda x, y: x < y
                self._best_val = np.inf
                self._best_iter = None
            case "max":
                self._criterion = lambda x, y: x > y
                self._best_val = -np.inf
                self._best_iter = None
            case _:
                raise ValueError(f"Invalid criterion: {criterion}")

    @property
    def best_value(self) -> float:
        return self._best_val

    @property
    def best_epoch(self) -> int:
        return self._best_iter

    @property
    def key(self):
        return self._key

    @key.setter
    def key(self, key: str):
        self.set_key(key)

    def set_key(self, key: str):
        """Change the metric key used to evaluate model quality."""
        match key:
            case "val_loss":

                def _val_loss(epoch_output: EpochOutput):
                    return epoch_output.val_loss  # type: ignore[return-value]

                self._get_val = _val_loss  # type: ignore
            case "train_loss":
                self._get_val = lambda epoch_output: epoch_output.train_loss
            case _:

                def _get_val(epoch_output: EpochOutput):
                    try:
                        return epoch_output.metrics[key]  # type: ignore
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

                self._get_val = _get_val

        self._key = key

    def on_epoch_end(
        self,
        model: Callable[..., Any],
        pbar: tqdm,
        epoch: int,
        epoch_output: EpochOutput,
        file_handler: FileHandler,
    ):
        """Evaluate the metric and optionally trigger a model save."""
        if epoch % self.val_every != 0 and self.key.startswith("val"):
            return

        new_val = self._get_val(epoch_output)
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
    """Save the best `flax.nnx` model state based on a given criterion.

    .. Note::
       This callback requires the `flax` and `orbax` packages to be installed.

    This callback is a subclass of :py:class:`trainax.callbacks.BestModelSaver`.
    It saves the best model state based on a given criterion. The saved
    model is defined by the `save_file` attribute.

    Parameters
    ----------
    save_path : str or Path
        Path to the directory where the best model will be saved.
    name : str, optional
        The name of the callback, by default "NNXBestModelSaver"
    key : Literal["train_loss", "val_loss"] or str, optional
        The key to use to save the best model, by default "val_loss"
    criterion : Literal["min", "max"] or Callable[[float, float], bool], optional
        Criterion used to determine best model. If a callable is provided,
        the function needs to be of the form
        f(new, old) -> bool[<new is better>]
    force_overwrite: bool, False
        Whether to overwrite an existing storage in case it already exists in
        `save_path`.


    Attributes
    ----------
    save_file : Path
        Path to the file where the best model will be saved.

    """

    save_file: Path

    def __init__(
        self,
        save_path: str | Path,
        name: str = "NNXBestModelSaver",
        key: Literal["train_loss", "val_loss"] | str = "val_loss",
        criterion: Literal["min", "max"] = "min",
        val_every: int = 1,
        force_overwrite: bool = False,
    ):
        try:
            from flax import nnx
        except ImportError as ie:
            raise ImportError(
                "NNXBestModelSaver requires flax (and orbax) to be installed."
            ) from ie

        try:
            import orbax.checkpoint as ocp
        except ImportError as ie:
            raise ImportError(
                "NNXBestModelSaver requires orbax to be installed"
            ) from ie

        super().__init__(self._save_model, name, key, criterion, val_every)
        self.save_file = Path(save_path) / "best_model"

        if self.save_file.exists():
            if not force_overwrite:
                raise ValueError(
                    f"Storage location {str(self.save_file)} already exists. "
                    "Either choose a different location or set "
                    "`force_overwrite=True` to overwrite the existing storage."
                )
            self.save_file.unlink()

    def _save_model(self, model, *args, **kwargs):
        import orbax.checkpoint as ocp
        from flax import nnx

        state = nnx.state(model)
        pure_dict_state = nnx.to_pure_dict(state)
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(self.save_file, pure_dict_state, force=True)

    @staticmethod
    def load_model(
        save_file: str | Path, model_cls, init_params: dict[str, Any] | None
    ):
        import orbax.checkpoint as ocp
        from flax import nnx

        with ocp.StandardCheckpointer() as checkpointer:
            restored_state = checkpointer.restore(save_file)

        if isinstance(model_cls, nnx.Module):
            return nnx.update(model_cls, restored_state)

        abstract_model = nnx.eval_shape(lambda: model_cls(**init_params))
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
        super().__init__(self._save_model, name, key, criterion)
        self.save_file = Path(save_path) / "best_model"

    @staticmethod
    def _save_model(model, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_model(
        save_file: str | Path, model_cls, init_params: dict[str, Any] | None
    ):
        raise NotImplementedError


class NNXMetricTracker(Callback):
    """Tracking metrics with nnx.metrics.Metric."""

    __slots__ = ["metrics", "history", "_mode"]

    def __init__(
        self,
        metrics,
        name: str = "NNXMetricTracker",
    ):
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
        # add data from single training step
        self.metrics.update(
            loss=step_output.loss, logits=step_output.yhat, labels=step_output.y
        )

    def on_val_end(self, epoch: int, data: list[ValStepOutput], **kwargs):
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
        # write metric to train history if no validation
        # else write metric to val history
        self._reset(epoch)
        # set to train mode
        self._mode = "train"

    def on_train_end(self, pbar: tqdm, **kwargs):
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
        return {k.split("_")[1] for k in self.history}

    def __getitem__(self, key: str) -> NDArray | tuple[NDArray, NDArray]:
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
        return self.plot_metric("loss", **kwargs)

    def plot_metric(self, metric: str, **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError as ie:
            raise ImportError(
                "NNXMetricTracker.plot_metric requires matplotlib to be "
                "installed."
            ) from ie

        ax = kwargs.pop("ax", None) or plt.figure(figsize=(8, 8)).add_subplot()
        ax.plot(*self[f"train_{metric}"].T, label=f"train {metric}", **kwargs)  # type: ignore

        try:
            ax.plot(*self[f"val_{metric}"].T, label=f"val {metric}", **kwargs)  # type: ignore
            ax.legend()
        except KeyError:
            pass

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())

        return ax
