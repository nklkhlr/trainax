import logging
from collections.abc import Callable
from typing import Any, Literal, TextIO

import numpy as np
from jaxtyping import Float, Int
from tqdm import tqdm

from trainax._file_handler import FileHandler
from trainax._types import EpochOutput, StepOutput


def _train_loss_msg(epoch_output: EpochOutput) -> str:
    """Format the training loss for human-readable logging."""
    return f"train loss={epoch_output.train_loss:.4E}"


class Callback:
    """Minimal base class for Trainax callbacks."""

    name: str

    def __init__(self, name: str):
        """Assign a unique `name` used for registry lookups."""
        self.name = name

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

    def on_step_end(self, pbar: tqdm, step_output: StepOutput):
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
        **kwargs,
    ):
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
        if isinstance(loss, float):
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
    _key: str
    _best_val: float
    _get_val: Callable[[EpochOutput], float]
    _criterion: Callable[[float, float], bool]

    def __init__(
        self,
        save_fun: Callable[[Callable[..., Any]], None],
        name: str = "BestModelSaver",
        key: Literal["train_loss", "val_loss"] | str = "val_loss",
        criterion: Literal["min", "max"] = "min",
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
        """
        super().__init__(name)

        self.save_model = save_fun
        self.set_key(key)

        match criterion:
            case "min":
                self._criterion = lambda x, y: x < y
                self._best_val = np.inf
            case "max":
                self._criterion = lambda x, y: x > y
                self._best_val = -np.inf
            case _:
                raise ValueError(f"Invalid criterion: {criterion}")

    @property
    def key(self):
        return self._key

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
        new_val = self._get_val(epoch_output)
        try:
            if self._criterion(new_val, self._best_val):
                self._best_val = new_val
                self.save_model(model)
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
