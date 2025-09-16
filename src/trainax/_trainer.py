from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.sharding as jsd
import numpy as np
from jaxtyping import Array, PyTree
from numpy.typing import NDArray
from optax import GradientTransformation
from tqdm import tqdm, trange
from tqdm.rich import tqdm as rich_tqdm
from tqdm.rich import trange as rich_trange

from trainax._callbacks import Callback
from trainax._dataloader import JaxLoader
from trainax._file_handler import FileHandler
from trainax._types import EpochOutput, PathLike, StepOutput, ValStepOutput

StepFun = Callable[[Callable, dict[str, NDArray | Array]], StepOutput]


class Trainer:
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
        return self._aggregate_steps

    def set_aggregate_steps(
        self, aggregate_steps: Literal["mean", "min", "max"]
    ):
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
            raise ValueError("Model sharding is not yet supported")

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
        self, epoch: int, val_step: StepFun, valloader: JaxLoader
    ) -> list[ValStepOutput]:
        step_results = []

        for callback in self.callbacks.values():
            callback.on_val_start(epoch=epoch, loader=valloader)

        if epoch % self.val_every == 0 and valloader is not None:
            for data in self._step_pbar(valloader, desc="Validation steps"):  # type: ignore
                # we check whether val_step is none when val data is given in
                # `train`
                step_results.append(val_step(data))  # type: ignore

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

    def update(self, model, optim: GradientTransformation, grads: PyTree, opt_state: PyTree):
        updates, opt_state = optim.update(grads, opt_state)
        # TODO: option to apply updates with other libraries
        model = eqx.apply_updates(model, updates)

    def train(
        self,
        model: Callable,
        optim: GradientTransformation,
        train_step: StepFun,
        trainloader: JaxLoader,
        val_step: StepFun | None,
        valloader: JaxLoader | None,
        jit_fun: Callable = eqx.filter_jit,
    ) -> Callable:
        agg_fun = self._agg_funs[self._aggregate_steps]
        if valloader is not None and val_step is None:
            raise ValueError(
                "'valloader' provided but val_step is not defined. "
                "Please set the validation step function "
                "(Trainer.val_step)."
            )

        self._prep_data(trainloader, valloader)

        @jit_fun
        def _update(model_, data_, opt_state_):
            out = train_step(model_, data_)
            jax.debug.print("step taken")
            updates, opt_state_ = optim.update(out.gradients, opt_state_)
            model_ = eqx.apply_updates(model_, updates)
            return model_, out, opt_state_

        for epoch in (epoch_bar := self._epoch_pbar()):
            for callback in self.callbacks.values():
                callback.on_epoch_start(
                    epoch=epoch,
                    pbar=epoch_bar,
                    file_handler=self.file_handler,
                )

            epoch_data: list[StepOutput] = []
            val_data: list[ValStepOutput] = []
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            for data in (step_bar := self._step_pbar(trainloader)):  # type: ignore
                print({key: type(val) for key, val in data.items()})
                model, output, opt_state = _update(model, data, opt_state)
                epoch_data.append(output)

                for callback in self.callbacks.values():
                    callback.on_step_end(pbar=step_bar, step_output=output)  # type: ignore

            if epoch % self.val_every and valloader is not None:
                val_data.append(
                    self._validation(epoch, jit_fun(val_step), valloader)  # type: ignore
                )

            epoch_output = EpochOutput.from_step_outputs(epoch_bar, agg_fun)  # type: ignore
            for callback in self.callbacks.values():
                callback.on_epoch_end(
                    model=model,
                    epoch=epoch,
                    pbar=epoch_bar,
                    epoch_output=epoch_output,
                    file_handler=self.file_handler,
                )

        return model