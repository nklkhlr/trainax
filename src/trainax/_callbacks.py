import logging
from collections.abc import Callable
from typing import Literal, TextIO

import numpy as np
from jaxtyping import Float, Int
from tqdm import tqdm

from trainax._file_handler import FileHandler
from trainax._types import EpochOutput, StepOutput


def _train_loss_msg(epoch_output: EpochOutput) -> str:
    return f"train loss={epoch_output.train_loss:.4E}"


class Callback:
    name: str

    def __init__(self, name: str):
        self.name = name

    # TODO: define interfaces for missing methods
    def on_epoch_start(self, *args, **kwargs):
        pass

    def on_epoch_end(
        self,
        model: Callable,
        pbar: tqdm,
        epoch: int,
        epoch_output: EpochOutput,
        file_handler: FileHandler,
    ):
        pass

    def on_step_end(self, pbar: tqdm, step_output: StepOutput):
        pass

    def on_val_start(self, *args, **kwargs):
        pass

    def on_val_end(self, *args, **kwargs):
        pass


class EpochLogger(Callback):
    logger: logging.Logger
    _val_message: Callable[[EpochOutput], str]
    _last_val_loss: float

    def __init__(
        self,
        logger: logging.Logger,
        name: str = "EpochLogger",
        has_validation: bool = True,
    ):
        super().__init__(name)
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
        msg = f"Epoch {epoch}: "
        self.logger.info(
            msg
            + _train_loss_msg(epoch_output)
            + self._val_message(epoch_output)
        )


class PbarHandler(Callback):
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
        msg = _train_loss_msg(epoch_output) + self._val_message(epoch_output)
        pbar.set_postfix_str(f"[{msg}]")


class LossMetricTracker(Callback):
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
    save_model: Callable
    _key: str
    _best_val: float
    _get_val: Callable[[EpochOutput], float]
    _criterion: Callable[[float, float], bool]

    def __init__(
        self,
        save_fun: Callable[[Callable]],
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
        match key:
            case "val_loss":
                self._get_val = lambda epoch_output: epoch_output.val_loss  # type: ignore
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
        model: Callable,
        pbar: tqdm,
        epoch: int,
        epoch_output: EpochOutput,
        file_handler: FileHandler,
    ):
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
