import logging
from typing import TextIO

from jaxtyping import Float, Int

from trainax._file_handler import FileHandler


class EpochLogger:
    logger: logging.Logger

    def __init__(self, logger):
        self.logger = logger

    def __call__(
        self, epoch: int, train_loss: float, vals_loss: float, **kwargs
    ):
        self.logger.info(
            f"Epoch {epoch}: "
            f"train loss={train_loss:.4f}, "
            f"(last) val loss={vals_loss:.4f}"
        )


class LossMetricTracker:
    _keys: dict[str, str]

    def __init__(self, train_file_key: str, val_file_key: str, **kwargs):
        self._keys = {"train": train_file_key, "val": val_file_key, **kwargs}

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

    def __call__(
        self,
        file_handler: FileHandler,
        train: Float | tuple[Int, Float],
        val: Float | tuple[Int, Float] | list[Int | Float] | None,
        **metrics,
    ):
        self._write_loss(file_handler[self._keys["train"]], train)
        if val is not None:
            self._write_loss(file_handler[self._keys["test"]], val)

        for key, value in metrics.items():
            self._write_loss(file_handler[key], value)
