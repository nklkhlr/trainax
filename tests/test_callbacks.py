import io
import logging
from pathlib import Path

import numpy as np
import pytest

from trainax._callbacks import (
    BestModelSaver,
    EpochLogger,
    LossMetricTracker,
    PbarHandler,
)
from trainax._file_handler import FileHandler
from trainax._types import EpochOutput


class DummyPbar:
    def __init__(self):
        self.postfix = None

    def set_postfix_str(self, text: str):
        self.postfix = text


def make_epoch_output(
    train_loss: float = 1.0,
    val_loss: float | None = 0.5,
    metrics: dict[str, float] | None = None,
) -> EpochOutput:
    return EpochOutput(
        train_loss=train_loss,
        train_losses=np.array([train_loss]),
        y=np.array([0.0]),
        yhat=np.array([0.0]),
        gradients=[],
        val_loss=val_loss,
        val_y=None if val_loss is None else np.array([0.0]),
        val_yhat=None if val_loss is None else np.array([0.0]),
        metrics=metrics,
    )


def test_epoch_logger_logs_with_and_without_validation(caplog):
    logger = logging.getLogger("epoch_logger_test")
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    callback = EpochLogger(logger=logger, has_validation=True)
    with caplog.at_level(logging.INFO):
        callback.on_epoch_end(
            model=None,
            pbar=None,
            epoch=1,
            epoch_output=make_epoch_output(val_loss=0.25),
            file_handler=None,  # type: ignore[arg-type]
        )

    assert "train loss=2.5000E-01" not in caplog.text
    assert "val loss=2.5000E-01" in caplog.text

    caplog.clear()
    callback_no_val = EpochLogger(logger=logger, has_validation=False)
    with caplog.at_level(logging.INFO):
        callback_no_val.on_epoch_end(
            model=None,
            pbar=None,
            epoch=1,
            epoch_output=make_epoch_output(val_loss=None),
            file_handler=None,  # type: ignore[arg-type]
        )
    assert "val loss" not in caplog.text


def test_pbar_handler_sets_postfix_for_val_and_train():
    pbar = DummyPbar()
    callback = PbarHandler(has_validation=True)
    callback.on_epoch_end(
        model=None,
        pbar=pbar,
        epoch=1,
        epoch_output=make_epoch_output(val_loss=0.1),
        file_handler=None,  # type: ignore[arg-type]
    )

    assert "train loss=1.0000E+00" in pbar.postfix
    assert "val loss=1.0000E-01" in pbar.postfix

    pbar_no_val = DummyPbar()
    callback_no_val = PbarHandler(has_validation=False)
    callback_no_val.on_epoch_end(
        model=None,
        pbar=pbar_no_val,
        epoch=1,
        epoch_output=make_epoch_output(val_loss=None),
        file_handler=None,  # type: ignore[arg-type]
    )
    assert "val loss" not in pbar_no_val.postfix


def test_loss_metric_tracker_writes_multiple_formats(tmp_path: Path):
    files = {
        "train_loss": tmp_path / "train.txt",
        "val_loss": tmp_path / "val.txt",
        "accuracy": tmp_path / "accuracy.txt",
    }
    tracker = LossMetricTracker()
    with FileHandler(files) as fh:
        epoch_output = make_epoch_output(
            train_loss=1.23,
            val_loss=0.45,
            metrics={"accuracy": 0.9},
        )
        tracker.on_epoch_end(
            model=None,
            pbar=None,  # type: ignore[arg-type]
            epoch=1,
            epoch_output=epoch_output,
            file_handler=fh,
        )

    assert tracker.losses["train_loss"] == [1.23, 0.45]
    assert (tmp_path / "train.txt").read_text().strip() == "1.23"
    assert (tmp_path / "val.txt").read_text().strip() == "0.45"
    assert (tmp_path / "accuracy.txt").read_text().strip() == "0.9"


def test_loss_metric_tracker_write_loss_errors():
    buffer = io.StringIO()
    with pytest.raises(ValueError, match="Invalid loss type"):
        LossMetricTracker._write_loss(buffer, {"bad": 1})


def test_best_model_saver_tracks_minimum():
    saved: list[int] = []

    def save_model(model, *args, **kwargs):
        saved.append(model)

    saver = BestModelSaver(save_model)
    fh = FileHandler({})

    saver.on_epoch_end(
        model=1,
        pbar=None,
        epoch=0,
        epoch_output=make_epoch_output(val_loss=0.5),
        file_handler=fh,
    )
    saver.on_epoch_end(
        model=2,
        pbar=None,
        epoch=1,
        epoch_output=make_epoch_output(val_loss=0.4),
        file_handler=fh,
    )
    saver.on_epoch_end(
        model=3,
        pbar=None,
        epoch=2,
        epoch_output=make_epoch_output(val_loss=0.6),
        file_handler=fh,
    )

    assert saved == [1, 2]


def test_best_model_saver_max_criterion():
    saved: list[int] = []

    def save_model(model, *args, **kwargs):
        saved.append(model)

    saver = BestModelSaver(save_model, criterion="max", key="train_loss")
    fh = FileHandler({})

    saver.on_epoch_end(
        model=1,
        pbar=None,
        epoch=0,
        epoch_output=make_epoch_output(train_loss=0.5, val_loss=None),
        file_handler=fh,
    )
    saver.on_epoch_end(
        model=2,
        pbar=None,
        epoch=1,
        epoch_output=make_epoch_output(train_loss=0.6, val_loss=None),
        file_handler=fh,
    )

    assert saved == [1, 2]


def test_best_model_saver_invalid_criterion():
    with pytest.raises(ValueError, match="Invalid criterion"):
        BestModelSaver(lambda _: None, criterion="median")


def test_best_model_saver_missing_val_loss_raises():
    saver = BestModelSaver(lambda _: None)
    fh = FileHandler({})

    with pytest.raises(
        ValueError,
        match="No value found for current epoch and key 'val_loss'",
    ):
        saver.on_epoch_end(
            model=None,
            pbar=None,
            epoch=0,
            epoch_output=make_epoch_output(val_loss=None),
            file_handler=fh,
        )


def test_best_model_saver_missing_metric_key():
    saver = BestModelSaver(lambda _: None, key="accuracy")
    fh = FileHandler({})

    with pytest.raises(ValueError, match="Invalid key: accuracy"):
        saver.on_epoch_end(
            model=None,
            pbar=None,
            epoch=0,
            epoch_output=make_epoch_output(metrics=None),
            file_handler=fh,
        )

    with pytest.raises(KeyError, match="Invalid key: accuracy"):
        saver.on_epoch_end(
            model=None,
            pbar=None,
            epoch=1,
            epoch_output=make_epoch_output(metrics={"precision": 0.8}),
            file_handler=fh,
        )


def test_best_model_saver_metric_key_success():
    saved: list[str] = []

    def save_model(model, *args, **kwargs):
        saved.append(model)

    saver = BestModelSaver(save_model, key="accuracy")
    fh = FileHandler({})

    saver.on_epoch_end(
        model="first",
        pbar=None,
        epoch=0,
        epoch_output=make_epoch_output(metrics={"accuracy": 0.7}),
        file_handler=fh,
    )
    saver.on_epoch_end(
        model="second",
        pbar=None,
        epoch=1,
        epoch_output=make_epoch_output(metrics={"accuracy": 0.8}),
        file_handler=fh,
    )

    assert saved == ["first"]
