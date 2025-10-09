from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.sharding as jsd
import jax.tree_util as jtu
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

StepFun = Callable[
    [Callable[..., Any], dict[str, NDArray | Array], PyTree | None],
    StepOutput,
]


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
        self,
        epoch: int,
        model: Callable[..., Any],
        val_step: StepFun,
        valloader: JaxLoader,
        val_state: PyTree | None,
    ) -> tuple[list[ValStepOutput], PyTree | None]:
        step_results: list[ValStepOutput] = []

        for callback in self.callbacks.values():
            callback.on_val_start(epoch=epoch, loader=valloader)

        if (epoch + 1) % self.val_every == 0 and valloader is not None:
            for data in self._step_pbar(  # type: ignore[arg-type]
                valloader, desc="Validation steps", leave=False
            ):
                state_input = self._clone_state(val_state)
                output = val_step(model, data, state_input)
                step_results.append(output)  # type: ignore[arg-type]
                if output.state is not None:
                    val_state = self._clone_state(output.state)

        for callback in self.callbacks.values():
            callback.on_val_end(epoch=epoch, data=step_results)

        return step_results, val_state

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
    def _clone_state(state: PyTree | None) -> PyTree | None:
        if state is None:
            return None

        def _clone_leaf(leaf):
            if isinstance(leaf, jax.Array):
                return jnp.array(leaf, copy=True)
            if isinstance(leaf, np.ndarray):
                return np.array(leaf, copy=True)
            return leaf

        leaves, treedef = jtu.tree_flatten(state)
        cloned_leaves = [_clone_leaf(leaf) for leaf in leaves]
        return jtu.tree_unflatten(treedef, cloned_leaves)

    def train(
        self,
        model: Callable[..., Any],
        optim: GradientTransformation,
        train_step: StepFun,
        trainloader: JaxLoader,
        val_step: StepFun | None,
        valloader: JaxLoader | None,
        train_state: PyTree | None = None,
        val_state: PyTree | None = None,
        jit_fun: Callable[
            [Callable[..., Any]],
            Callable[..., Any],
        ] = eqx.filter_jit,
    ) -> tuple[Callable[..., Any], PyTree | None]:
        agg_fun = self._agg_funs[self._aggregate_steps]
        if valloader is not None and val_step is None:
            raise ValueError(
                "'valloader' provided but val_step is not defined. "
                "Please set the validation step function "
                "(Trainer.val_step)."
            )

        trainloader, valloader = self._prep_data(trainloader, valloader)

        def _update(
            model_: Callable[..., Any],
            data_: dict[str, NDArray | Array],
            opt_state_: PyTree,
            state_: PyTree | None,
        ):
            state_input = self._clone_state(state_)
            out = train_step(model_, data_, state_input)
            if out.gradients is None:
                raise ValueError(
                    "train_step must return gradients to apply optimizer "
                    "updates."
                )
            updates, opt_state_new = optim.update(out.gradients, opt_state_)
            model_new = eqx.apply_updates(model_, updates)
            state_new = (
                self._clone_state(out.state)
                if out.state is not None
                else state_
            )
            return model_new, out, opt_state_new, state_new

        update_fun = jit_fun(_update)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        state = train_state

        val_step_fun: StepFun | None = None
        if valloader is not None and val_step is not None:
            val_step_fun = jit_fun(val_step)

        for epoch in (epoch_bar := self._epoch_pbar()):
            for callback in self.callbacks.values():
                callback.on_epoch_start(
                    epoch=epoch,
                    pbar=epoch_bar,
                    file_handler=self.file_handler,
                )

            epoch_data: list[StepOutput] = []
            step_bar = self._step_pbar(trainloader)  # type: ignore[arg-type]
            for data in step_bar:
                model, output, opt_state, state = update_fun(
                    model, data, opt_state, state
                )
                epoch_data.append(output)

                for callback in self.callbacks.values():
                    callback.on_step_end(
                        pbar=step_bar,
                        step_output=output,
                    )

            val_outputs: list[ValStepOutput] | None = None
            if valloader is not None and val_step_fun is not None:
                val_outputs, val_state = self._validation(
                    epoch=epoch,
                    model=model,
                    val_step=val_step_fun,
                    valloader=valloader,
                    val_state=val_state,
                )
                if not val_outputs:
                    val_outputs = None

            epoch_output = EpochOutput.from_step_outputs(
                epoch_data,
                agg_fun,
                val_outputs,
            )
            for callback in self.callbacks.values():
                callback.on_epoch_end(
                    model=model,
                    epoch=epoch,
                    pbar=epoch_bar,
                    epoch_output=epoch_output,
                    file_handler=self.file_handler,
                )

        return model, state
