import warnings
from collections.abc import Callable, Generator

import jax
import jax.sharding as jsd
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array
from numpy.typing import NDArray


class JaxLoader:
    """JaxLoader is a class for loading data in batches.

    Data is stored as and provided a dictionary of jax/numpy arrays
    """

    _data: dict[str, Array | NDArray]
    _batch_size: int
    _n_batches: int
    _sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None
    _batch_func: Callable
    _rng: np.random.Generator

    def __init__(
        self,
        data: dict,
        batch_size: int,
        sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None = None,
        seed: int = 42,
    ):
        # TODO: type checks for data and batch_size
        """
        Initialize JaxLoader with data, batch size, sharding, and random seed.

        Parameters
        ----------
        data : dict
            Dictionary containing data to be loaded. Values should be jax or
            numpy arrays. Main data array is expected to named "x".
        batch_size : int
            Batch size to be used for splitting data.
        sharding : jsd.NamedSharding | jsd.SingleDeviceSharding | None
            Sharding configuration to be used for data loading.
        seed : int
            Random seed to be used for data loading.

        Notes
        -----
        If the nubmer of samples (i.e. data["x"].shape[0]) is not divisible by
        the number of samples per batch (i.e. batch_size), the last batch will
        be dropped.
        """
        if not "x" in data:
            raise ValueError(
                "'data' must contain a key 'x'. This should be the main "
                "data array and the shape along the first axis should "
                "correspond to the number of samples."
            )

        self._data = data
        self._rng = np.random.default_rng(seed or seed)

        # sets _batch_size and _n_batches
        self._set_batch_size(batch_size)
        # sets _sharding and self._batch_func
        self._set_sharding(sharding)

    def _set_batch_size(self, batch_size):
        self._batch_size = batch_size

        if self._batch_size > len(self._data["x"]):
            warnings.warn(
                f"Batch size {batch_size} is larger than dataset size "
                f"{len(self._data['x'])}. Setting batch size to dataset size.",
                UserWarning,
                stacklevel=2,
            )
            self._batch_size = len(self._data["x"])
        else:
            self._n_batches = len(self._data["x"]) // batch_size
            if len(self._data["x"]) % batch_size != 0:
                warnings.warn(
                    "Batch size is not a multiple of dataset size. "
                    "Dropping last batch (random samples for each epoch)",
                    UserWarning,
                    stacklevel=2,
                )

    @property
    def batch_size(self) -> int:
        """Get or set batch size of the loader.

        Returns
        -------
        int
            Batch size of the loader.

        Notes
        -----
        If the batch size is set to a value larger than the dataset size, the
        batch size will be set to the dataset size. If the dataset size is not
        a multiple of the batch size, the last batch will be dropped.

        Parameters
        ----------
        batch_size : int
            Batch size to be set.

        Examples
        --------
        >>> loader = JaxLoader(data={"x": jnp.arange(10)}, batch_size=3)
        >>> loader.batch_size
        3
        """
        return self._batch_size

    def set_batch_size(self, batch_size: int):
        self._set_batch_size(batch_size)

    @property
    def n_batches(self) -> int:
        """Number of batches in the loader."""
        return self._n_batches

    @property
    def n_points(self) -> int:
        """Number of points in the dataset."""
        return self._data["x"].shape[0]

    def _set_sharding(
        self, sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None
    ):
        self._sharding = sharding
        if sharding is None:
            self._batch_func = self._get_batch
        else:
            n_shards = sharding.num_devices
            assert self._batch_size % n_shards == 0, (
                f"Batch size {self._batch_size} is not a multiple of the "
                f"number of shards {n_shards}"
            )
            self._batch_func = self._shard_batch

    @property
    def sharding(self) -> jsd.NamedSharding | jsd.SingleDeviceSharding | None:
        """Sharding configuration used for data loading.

        If `sharding` is `None`, no sharding is used. If `sharding` is
        `jsd.NamedSharding`, the data is sharded across the specified devices.
        If `sharding` is `jsd.SingleDeviceSharding`, the data is sharded onto a
        single device.

        Returns
        -------
        sharding : jsd.NamedSharding | jsd.SingleDeviceSharding | None
            Sharding configuration used for data loading.
        """
        return self._sharding

    def set_sharding(
        self, sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None
    ):
        """Set the sharding configuration for the data loader.

        Parameters
        ----------
        sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None
            Sharding configuration to set
        """
        self._set_sharding(sharding)

    def _get_batch(
        self, indices: NDArray, idx: int
    ) -> dict[str, Array | NDArray]:
        idxs = indices[idx * self._batch_size : (idx + 1) * self._batch_size]
        return {key: self._data[key][idxs, ...] for key in self._data}

    def _shard_batch(
        self, indices: NDArray, idx: int
    ) -> dict[str, Array | NDArray]:
        batch = self._get_batch(indices, idx)
        return {
            key: jax.make_array_from_process_local_data(self._sharding, arr)
            for key, arr in batch.items()
        }

    def __getitem__(self, idx: int | slice) -> dict[str, Array | NDArray]:
        return {key: arr[idx, ...] for key, arr in self._data.items()}

    def __iter__(self) -> Generator[dict[str, Array | NDArray]]:
        indices = np.arange(self.n_points)
        self._rng.shuffle(indices)
        for idx in range(self._n_batches):
            yield self._batch_func(indices, idx)

    def __len__(self) -> int:
        return self._n_batches

    def __repr__(self) -> str:
        data_keys = ", ".join(list(self._data.keys()))
        if self._sharding is None:
            sharding_note = "No sharding"
        elif isinstance(self._sharding, jsd.NamedSharding):
            sharding_note = (
                f"Sharding: {self._sharding.num_devices} devices along "
                f"{self._sharding.mesh.axis_names}"
            )
        else:
            sharding_note = "Sharding: single device"

        return (
            f"{self.__class__.__name__}(Data attributes: {data_keys}, "
            f"Batch size: {self._batch_size}, "
            f"N batches: {self._n_batches}, "
            f"Sharding: {sharding_note})"
        )

    def flatten_with_keys(self):
        children = [jtu.GetAttrKey("_data"), self._data]
        aux_data = (self._batch_size, self._sharding, self._rng)
        return children, aux_data

    def flatten(self):
        children = [self._data]
        aux_data = (self._batch_size, self._sharding, self._rng)
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        new = cls(
            data=children[0],
            batch_size=aux_data[0],
            sharding=aux_data[2],
        )
        new._rng = aux_data[2]
        return new


class SingleBatchJaxLoader(JaxLoader):
    _points_per_shard: int

    def __init__(
        self,
        data: dict,
        sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None = None,
        seed: int = 42,
    ):
        super().__init__(data=data, batch_size=1, sharding=sharding, seed=seed)

    def _set_batch_size(self, batch_size):
        if batch_size != 1:
            warnings.warn(
                "Batch size must be 1 for SingleBatchJaxLoader",
                stacklevel=1,
                category=UserWarning,
            )
        self._batch_size = self.n_points

    def _set_sharding(
        self, sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None
    ):
        self._sharding = sharding
        if sharding is None:
            self._batch_func = self._get_batch
        else:
            # we don't want to required the number of points to be divisible
            # by the number of shards (too restrictive)
            rem_shard = self.n_points % sharding.num_devices
            if rem_shard == 0:
                self._points_per_shard = self.n_points
            else:
                warnings.warn(
                    f"Number of points {self.n_points} is not divisible by "
                    f"the number of shards {sharding.num_devices}. "
                    f"Remove the last {rem_shard} points per epoch at random.",
                    UserWarning,
                    stacklevel=2,
                )
                self._points_per_shard = self.n_points - rem_shard

            self._batch_func = self._shard_batch

    def _shard_batch(
        self, indices: NDArray, idx: int
    ) -> dict[str, Array | NDArray]:
        batch = self._get_batch(indices[: self._points_per_shard], idx)
        return {
            key: jax.make_array_from_process_local_data(self._sharding, arr)
            for key, arr in batch.items()
        }

    def __len__(self) -> int:
        return 1

    def flatten_with_keys(self):  # pyright: ignore
        children = (jtu.GetAttrKey("_data"), self._data)
        aux_data = (self._sharding, self._rng, self._points_per_shard)
        return children, aux_data

    def flatten(self):  # pyright: ignore
        children = self._data
        aux_data = (self._sharding, self._rng, self._points_per_shard)
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        new = cls(
            data=children[0],
            sharding=aux_data[0],
        )
        new._rng = aux_data[1]
        new._points_per_shard = aux_data[2]
        return new


jtu.register_pytree_with_keys(
    JaxLoader,
    JaxLoader.flatten_with_keys,  # pyright: ignore
    JaxLoader.unflatten,  # pyright: ignore
    JaxLoader.flatten,  # pyright: ignore
)
jtu.register_pytree_with_keys(
    SingleBatchJaxLoader,
    SingleBatchJaxLoader.flatten_with_keys,  # pyright: ignore
    SingleBatchJaxLoader.unflatten,  # pyright: ignore
    SingleBatchJaxLoader.flatten,  # pyright: ignore
)
