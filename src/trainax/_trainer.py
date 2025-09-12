from collections.abc import Callable
from typing import Literal

import jax
import jax.sharding as jsd
import numpy as np
from tqdm import tqdm, trange
from tqdm.rich import tqdm as rich_tqdm
from tqdm.rich import trange as rich_trange

from trainax._dataloader import JaxLoader

StepFun = Callable[[dict], dict]


class Trainer:
    make_step: StepFun
    val_step: StepFun | None
    epoch_callbacks: dict
    step_callbacks: dict
    logger_callbacks: dict

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
        make_step: StepFun,
        epoch_callbacks: list,
        step_callbacks: list,
        logger_callbacks: list,
        file_handlers: list,
        n_epochs: int,
        val_step: StepFun | None = None,
        val_every: int = 5,
        use_rich: bool = True,
        model_sharding: list[int] | int | jsd.NamedSharding | None = None,
        data_sharding: list[int] | int | jsd.NamedSharding | None = None,
        aggregate_steps: Literal["mean", "min", "max"] = "mean",
    ):
        self.make_step = make_step
        self.val_step = val_step
        # TODO: unify callbacks into one list similar to torch lightning
        self.epoch_callbacks = {
            callback.__name__: callback for callback in epoch_callbacks
        }
        self.step_callbacks = {
            callback.__name__: callback for callback in step_callbacks
        }
        self.logger_callbacks = {
            callback.__name__: callback for callback in logger_callbacks
        }
        self.file_handlers = file_handlers
        self.n_epochs = n_epochs
        self.val_every = val_every
        self.use_rich = use_rich

        self._sharding = {}
        self._set_sharding(model_sharding, "model")
        self._set_sharding(data_sharding, "data")

    @property
    def aggregate_steps(self):
        return self._aggregate_steps

    def set_aggregate_steps(
        self, aggregate_steps: Literal["mean", "min", "max"]
    ):
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
        return self._sharding

    def set_sharding(self, sharding, kind: Literal["data", "model"]):
        # TODO: note that if sharding is int or list[int] only single dimension
        # sharding is supported
        self._set_sharding(sharding, kind)

    def _epoch_pbar(self, **kwargs) -> tqdm | rich_tqdm:
        range_fun = rich_trange if self.use_rich else trange
        desc = "Training epochs"
        return range_fun(
            self.n_epochs,
            desc=desc,
            bar_format=(
                "{desc} [{n:d}/{total_fmt} ({percentage:3.0f}%)] | "
                "{bar} [{elapsed}<{remaining}, {rate_fmt}] | {postfix}"
            ),
            **kwargs,
        )

    def _step_pbar(
        self, loader: JaxLoader, **kwargs
    ) -> tqdm | rich_tqdm | JaxLoader:
        if len(loader) > 1:
            tqdm_fun = rich_tqdm if self.use_rich else tqdm
            return tqdm_fun(
                loader,  # type: ignore
                desc=kwargs.pop("desc", "Epoch steps"),
                total=len(loader),
                bar_format=(
                    "{desc} [{n:d}/{total_fmt} | {percentage:3.0f}%] | "
                    "{bar} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ),
                leave=kwargs.pop("leave", False),
                **kwargs,
            )

        return loader

    def _validation(
        self, epoch: int, valloader: JaxLoader
    ) -> dict[str, np.ndarray]:
        step_results = []
        if epoch % self.val_every == 0 and valloader is not None:
            for data in self._step_pbar(valloader, desc="Validation"):  # type: ignore
                step_results.append(self.val_step(data))

        # TODO: turn list[dict[str, float]] into dict[str, Float[np.ndarray, ""]]
        return step_results

    def _prep_data(
        self, trainloader: JaxLoader, valloader: JaxLoader | None
    ) -> tuple[JaxLoader, JaxLoader | None]:
        if (sharding := self._sharding.get("data")) is not None:
            trainloader.set_sharding(sharding)
            if valloader is not None:
                valloader.set_sharding(sharding)

        return trainloader, valloader

    def train(self, trainloader: JaxLoader, valloader: JaxLoader | None):
        agg_fun = self._agg_funs[self._aggregate_steps]
        if valloader is not None:
            if self.val_step is None:
                raise ValueError(
                    "'valloader' provided but val_step is not defined. "
                    "Please set the validation step function "
                    "(Trainer.val_step)."
                )

            def val_func(epoch: int, valloader: JaxLoader) -> dict[str, float]:  # type: ignore
                val_metrics = self._validation(epoch, valloader)
                # TODO: aggregate over each entry in val_metrics
                return {
                    key: agg_fun(metric) for key, metric in val_metrics.items()
                }
        else:

            def val_func(*args, **kwargs):
                pass

        for epoch in (epoch_bar := self._epoch_pbar()):
            epoch_data = []
            for data in (step_bar := self._step_pbar(trainloader)):  # type: ignore
                output = self.make_step(data)
                epoch_data.append(output)
                for callback in self.step_callbacks.values():
                    callback(pbar=step_bar, output=output)

            for callback in self.epoch_callbacks.values():
                callback(epoch=epoch, data=epoch_data, pbar=epoch_bar)

            for callback in self.logger_callbacks.values():
                callback(epoch=epoch, callbacks=self.epoch_callbacks)
