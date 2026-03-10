import warnings
from collections.abc import Callable, Generator
from typing import TypeVar

import jax
import jax.sharding as jsd
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array
from numpy.typing import NDArray

T = TypeVar("T", Array, NDArray)


class JaxLoader[T]:
    """Lightweight data loader for in-memory numpy/jax datasets.

    Data is stored as a dictionary of jax/numpy arrays and yielded in
    shuffled batches each epoch. The loader can be registered as a JAX pytree
    via :func:`jax.tree_util.register_pytree_with_keys`.

    Attributes
    ----------
    batch_size : int
        Number of samples per batch (property, settable).
    seed : int
        Random seed controlling shuffle order (property, settable).
    n_batches : int
        Number of complete batches per epoch (property).
    n_points : int
        Total number of samples in the dataset (property).
    data : dict[str, T]
        The underlying data dictionary (property).
    sharding : jsd.NamedSharding | jsd.SingleDeviceSharding | None
        Active JAX sharding configuration (property, settable).

    Methods
    -------
    set_batch_size(batch_size)
        Update the batch size.
    set_sharding(sharding)
        Update the sharding configuration.
    __getitem__(idx)
        Return raw data by index or slice.
    __iter__()
        Iterate over shuffled batches.
    __len__()
        Return the number of batches per epoch.
    """

    _data: dict[str, T]
    _batch_size: int
    _n_batches: int
    _sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None
    _batch_func: Callable
    _rng: np.random.Generator
    _seed: int
    _x_key: str

    def __init__(
        self,
        data: dict,
        batch_size: int,
        sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None = None,
        seed: int = 42,
        x_key: str = "x",
    ):
        # TODO: type checks for data and batch_size
        """Initialize a new JaxLoader instance.

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
        x_key : str
            Name of the main data array. Used to determine the number of points
            in the dataset.

        Notes
        -----
        If the number of samples (i.e. data[x_key].shape[0]) is not divisible by
        the number of samples per batch (i.e. batch_size), the last batch will
        be dropped.

        Examples
        --------
        >>> import numpy as np
        >>> x = np.random.randn(100, 10).astype(np.float32)
        >>> y = np.random.randn(100, 1).astype(np.float32)
        >>> loader = JaxLoader({"x": x, "y": y}, batch_size=32)
        >>> len(loader)  # 100 // 32 = 3 complete batches
        3
        >>> for batch in loader:
        ...     print(batch["x"].shape)
        (32, 10)
        (32, 10)
        (32, 10)
        """
        if x_key not in data:
            raise ValueError(
                f"'data' must contain a key '{x_key}'. This should be the main "
                "data array and the shape along the first axis should "
                "correspond to the number of samples."
            )
        self._x_key = x_key

        self._data = data
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # sets _batch_size and _n_batches
        self._set_batch_size(batch_size)
        # sets _sharding and self._batch_func
        self._set_sharding(sharding)

    def _set_batch_size(self, batch_size):
        self._batch_size = batch_size

        if self._batch_size > self.n_points:
            warnings.warn(
                f"Batch size ({batch_size}) is larger than dataset size "
                f"{self.n_points}. Setting batch size to "
                "dataset size.",
                UserWarning,
                stacklevel=2,
            )
            self._batch_size = self.n_points
            self._n_batches = 1
        else:
            self._n_batches = self.n_points // batch_size
            if self.n_points % batch_size != 0:
                warnings.warn(
                    f"Batch size ({batch_size}) is not a multiple of dataset "
                    f"size {self.n_points}. Dropping last batch "
                    "(random samples for each epoch)",
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

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._set_batch_size(batch_size)

    def set_batch_size(self, batch_size: int):
        """Set a new batch size for the loader.

        Parameters
        ----------
        batch_size : int
            Desired number of samples per batch. Values larger than the dataset
            size are clipped to the dataset length, while non-divisible sizes
            trigger a warning and drop the remainder each epoch.
        """
        self._set_batch_size(batch_size)

    @property
    def seed(self) -> int:
        """Seed used for random splitting."""
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        """Set a new random seed and reinitialise the RNG.

        Parameters
        ----------
        seed : int
            New random seed for the shuffle generator.
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def n_batches(self) -> int:
        """Number of batches in the loader."""
        return self._n_batches

    @property
    def n_points(self) -> int:
        """Number of points in the dataset."""
        return self._data[self._x_key].shape[0]  # type: ignore

    @property
    def data(self) -> dict[str, T]:
        """Get the data dictionary."""
        return self._data

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

    @sharding.setter
    def sharding(self, sharding: jsd.NamedSharding | jsd.SingleDeviceSharding):
        self._set_sharding(sharding)

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
        return {key: self._data[key][idxs, ...] for key in self._data}  # type: ignore

    def _shard_batch(
        self, indices: NDArray, idx: int
    ) -> dict[str, Array | NDArray]:
        batch = self._get_batch(indices, idx)
        return {
            key: jax.make_array_from_process_local_data(self._sharding, arr)
            for key, arr in batch.items()
        }

    def __getitem__(self, idx: int | slice) -> dict[str, T]:
        """Return raw data by index or slice without shuffling.

        Parameters
        ----------
        idx : int | slice
            Index or slice to extract along the batch axis.

        Returns
        -------
        dict[str, Array | NDArray]
            Dictionary mapping feature names to array slices.
        """
        return {key: arr[idx, ...] for key, arr in self._data.items()}  # type: ignore

    def __iter__(self) -> Generator[dict[str, T]]:
        """Iterate over shuffled batches for the current configuration."""
        indices = np.arange(self.n_points)
        self._rng.shuffle(indices)
        for idx in range(self._n_batches):
            yield self._batch_func(indices, idx)

    def __len__(self) -> int:
        """Return the number of batches produced per epoch."""
        return self._n_batches

    def __repr__(self) -> str:
        """Return a concise summary of the loader configuration."""
        data_keys = list(self._data.keys())
        if self._sharding is None:
            sharding_note = "None"
        elif isinstance(self._sharding, jsd.NamedSharding):
            sharding_note = (
                f"Sharding: {self._sharding.num_devices} devices along "
                f"{self._sharding.mesh.axis_names}"
            )
        else:
            sharding_note = "Sharding: single device"

        return (
            f"{self.__class__.__name__}(Data attributes: {data_keys} | "
            f"N samples: {self.n_points} | "
            f"Batch size: {self._batch_size} | "
            f"N batches: {self._n_batches} | "
            f"Sharding: {sharding_note})"
        )

    @property
    def data_type(self) -> type:
        """Return the data type of the main data array."""
        return type(self._data[self._x_key])

    def flatten_with_keys(self):
        """Return pytree metadata for registration with ``jax.tree_util``.

        Returns
        -------
        tuple[list, tuple]
            A ``(children, aux_data)`` pair where children includes attribute
            keys for use with :func:`jax.tree_util.register_pytree_with_keys`.
        """
        children = [jtu.GetAttrKey("_data"), self._data]
        aux_data = (self._batch_size, self._sharding, self._rng)
        return children, aux_data

    def flatten(self):
        """Return pytree children/aux data without attribute keys.

        Returns
        -------
        tuple[list, tuple]
            A ``(children, aux_data)`` pair for standard pytree registration.
        """
        children = [self._data]
        aux_data = (self._batch_size, self._sharding, self._rng)
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        """Reconstruct a loader instance from pytree children.

        Parameters
        ----------
        aux_data : tuple
            Auxiliary data produced by :meth:`flatten` containing
            ``(batch_size, sharding, rng)``.
        children : list
            Pytree children produced by :meth:`flatten` containing the data
            dict.

        Returns
        -------
        JaxLoader
            Reconstructed loader instance.
        """
        new = cls(
            data=children[0],
            batch_size=aux_data[0],
            sharding=aux_data[2],
        )
        new._rng = aux_data[2]
        return new


class SingleBatchJaxLoader[T](JaxLoader):
    """Specialisation of :class:`JaxLoader` that always yields one batch.

    The entire dataset is returned as a single batch each epoch. Useful for
    small in-memory datasets where no batching is required. Sharding is still
    supported; data is split across devices without requiring the dataset size
    to be divisible by the number of shards.

    Methods
    -------
    __len__()
        Always returns 1.
    flatten_with_keys()
        Return pytree representation with attribute keys.
    flatten()
        Return pytree representation without attribute keys.
    unflatten(aux_data, children)
        Reconstruct a single-batch loader from pytree children.
    """

    _points_per_shard: int
    _n_batches: int = 1

    def __init__(
        self,
        data: dict,
        sharding: jsd.NamedSharding | jsd.SingleDeviceSharding | None = None,
        seed: int = 42,
        **kwargs,
    ):
        """Initialise a single-batch data loader.

        Parameters
        ----------
        data : dict
            Dictionary of jax/numpy arrays. Must contain the key specified by
            ``x_key`` (default ``"x"``) to determine the dataset size.
        sharding : jsd.NamedSharding | jsd.SingleDeviceSharding | None, None
            Optional JAX sharding configuration.
        seed : int, 42
            Random seed (used by the parent class; shuffle order does not
            affect single-batch loading).
        **kwargs
            Additional keyword arguments forwarded to :class:`JaxLoader`
            (e.g. ``x_key``).

        Returns
        -------
        None

        Notes
        -----
        The ``batch_size`` parameter is fixed to the dataset length; passing
        any other value emits a :class:`UserWarning` and is ignored.
        """
        super().__init__(
            data=data, batch_size=1, sharding=sharding, seed=seed, **kwargs
        )

    def _get_batch(self, indices: NDArray, idx: int) -> dict[str, T]:  # type: ignore[override]
        return {key: self._data[key] for key in self._data}

    def _set_batch_size(self, batch_size):
        """Force the internal batch size to match the dataset length."""
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

    def _shard_batch(self, indices: NDArray, idx: int) -> dict[str, T]:  # type: ignore[override]
        batch = self._get_batch(indices[: self._points_per_shard], idx)
        return {
            key: jax.make_array_from_process_local_data(self._sharding, arr)
            for key, arr in batch.items()
        }

    def __len__(self) -> int:
        """Return the number of batches per epoch (always 1).

        Returns
        -------
        int
            Always ``1``.
        """
        return 1

    def flatten_with_keys(self):  # pyright: ignore
        """Return pytree representation with attribute keys.

        Returns
        -------
        tuple[tuple, tuple]
            A ``(children, aux_data)`` pair with attribute keys for use with
            :func:`jax.tree_util.register_pytree_with_keys`.
        """
        children = (jtu.GetAttrKey("_data"), self._data)
        aux_data = (self._sharding, self._rng, self._points_per_shard)
        return children, aux_data

    def flatten(self):  # pyright: ignore
        """Return pytree representation without attribute keys.

        Returns
        -------
        tuple[Any, tuple]
            A ``(children, aux_data)`` pair for standard pytree registration.
        """
        children = self._data
        aux_data = (self._sharding, self._rng, self._points_per_shard)
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        """Reconstruct a single-batch loader from pytree children.

        Parameters
        ----------
        aux_data : tuple
            Auxiliary data from :meth:`flatten` containing
            ``(sharding, rng, points_per_shard)``.
        children : Any
            Data dict produced by :meth:`flatten`.

        Returns
        -------
        SingleBatchJaxLoader
            Reconstructed loader instance.
        """
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
