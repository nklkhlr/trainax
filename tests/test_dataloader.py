import jax
import jax.sharding as jsd
import numpy as np
import pytest
from jax.sharding import PartitionSpec

import trainax._dataloader as dataloader_module
from trainax._dataloader import JaxLoader, SingleBatchJaxLoader

jax.config.update("jax_num_cpu_devices", 2)


@pytest.fixture
def sample_data():
    def _factory(
        n: int = 12,
        m: int = 3,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, m)).astype(np.float32)
        y = np.sin(x)
        return {"x": x, "y": y}

    return _factory


@pytest.fixture
def single_device_sharding():
    devices = jax.devices()
    if not devices:
        pytest.skip("No JAX devices available for sharding tests.")
    return jsd.SingleDeviceSharding(devices[0])


@pytest.fixture
def named_sharding():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("Need at least two devices for NamedSharding tests.")
    mesh = jsd.Mesh(np.array(devices[:2]), ("data",))
    return jsd.NamedSharding(mesh, PartitionSpec("data"))


def test_loader_requires_main_input(sample_data):
    bad_data = {"y": sample_data(n=3, m=2)["y"]}
    with pytest.raises(ValueError, match="must contain a key 'x'"):
        JaxLoader(data=bad_data, batch_size=1)


def test_loader_basic_properties(sample_data):
    loader = JaxLoader(
        data=sample_data(n=12, m=4, seed=1),
        batch_size=3,
        sharding=None,
        seed=0,
    )
    assert loader.n_points == 12
    assert loader.batch_size == 3
    assert loader.n_batches == 4
    assert len(loader) == 4
    assert loader.sharding is None


@pytest.mark.parametrize(
    ("indexer", "expected_shape"),
    [
        (0, (4,)),
        (slice(0, 2), (2, 4)),
    ],
)
def test_loader_getitem_returns_batches(sample_data, indexer, expected_shape):
    loader = JaxLoader(
        data=sample_data(n=6, m=4, seed=5),
        batch_size=3,
        sharding=None,
        seed=0,
    )
    batch = loader[indexer]
    assert set(batch.keys()) == {"x", "y"}
    assert batch["x"].shape == expected_shape
    assert batch["y"].shape == expected_shape


def test_loader_iteration_drops_last_batch(sample_data):
    with pytest.warns(UserWarning, match="Dropping last batch"):
        loader = JaxLoader(
            data=sample_data(n=10, m=3, seed=2),
            batch_size=4,
            sharding=None,
            seed=0,
        )

    batches = list(loader)
    assert len(batches) == loader.n_batches == 2
    for batch in batches:
        assert batch["x"].shape[0] == loader.batch_size
        assert batch["y"].shape[0] == loader.batch_size

    combined = np.concatenate([batch["x"] for batch in batches], axis=0)
    assert combined.shape == (loader.n_batches * loader.batch_size, 3)


@pytest.mark.parametrize(
    ("new_batch_size", "warn_match", "expected_batches"),
    [
        (7, "Dropping last batch", 1),
        (50, "larger than dataset size", None),
    ],
)
def test_loader_set_batch_size_warnings(
    sample_data, new_batch_size, warn_match, expected_batches
):
    loader = JaxLoader(
        data=sample_data(n=8, m=2, seed=3),
        batch_size=4,
        sharding=None,
        seed=0,
    )
    with pytest.warns(UserWarning, match=warn_match):
        loader.set_batch_size(new_batch_size)

    if expected_batches is not None:
        assert loader.n_batches == expected_batches
    assert loader.batch_size <= loader.n_points


def test_loader_set_sharding_requires_multiple(sample_data, named_sharding):
    loader = JaxLoader(
        data=sample_data(n=6, m=2, seed=4),
        batch_size=3,
        sharding=None,
        seed=0,
    )
    with pytest.raises(
        AssertionError,
        match="Batch size 3.*number of shards 2",
    ):
        loader.set_sharding(named_sharding)


def test_loader_iter_uses_sharded_batches(
    sample_data, monkeypatch, single_device_sharding
):
    loader = JaxLoader(
        data=sample_data(n=9, m=2, seed=6),
        batch_size=3,
        sharding=None,
        seed=0,
    )
    loader.set_sharding(single_device_sharding)

    calls: list[tuple[jsd.Sharding, tuple[int, int]]] = []

    def fake_make_array(sharding, arr):
        calls.append((sharding, arr.shape))
        return arr

    monkeypatch.setattr(
        dataloader_module.jax,
        "make_array_from_process_local_data",
        fake_make_array,
    )

    batches = list(loader)
    assert len(batches) == loader.n_batches
    assert calls
    assert all(call[0] is single_device_sharding for call in calls)
    assert all(shape[0] == loader.batch_size for _, shape in calls)


def test_loader_repr_contains_summary(sample_data):
    loader = JaxLoader(
        data=sample_data(n=4, m=2, seed=7),
        batch_size=2,
        sharding=None,
        seed=0,
    )
    representation = repr(loader)
    assert "JaxLoader" in representation
    assert "Data attributes: ['x', 'y']" in representation
    assert "Batch size: 2" in representation


def test_loader_flatten_helpers(sample_data):
    loader = JaxLoader(
        data=sample_data(n=4, m=2, seed=8),
        batch_size=2,
        sharding=None,
        seed=0,
    )
    children, aux = loader.flatten()
    assert children[0] is loader._data  # noqa: SLF001
    assert aux[0] == loader.batch_size

    keyed_children, keyed_aux = loader.flatten_with_keys()
    assert keyed_aux[0] == loader.batch_size
    assert keyed_children[1] is loader._data  # noqa: SLF001

    restored = JaxLoader.unflatten((aux[0], None, None), children)
    assert isinstance(restored, JaxLoader)
    assert restored.batch_size == aux[0]
    assert restored.n_points == loader.n_points


def test_single_batch_loader_len_and_batch_size(sample_data):
    loader = SingleBatchJaxLoader(
        data=sample_data(n=5, m=3, seed=9),
        sharding=None,
        seed=0,
    )
    assert loader.batch_size == loader.n_points
    assert len(loader) == 1


def test_single_batch_loader_rejects_manual_batch_size(sample_data):
    loader = SingleBatchJaxLoader(
        data=sample_data(n=6, m=2, seed=10),
        sharding=None,
        seed=0,
    )
    with pytest.warns(UserWarning, match="Batch size must be 1"):
        loader.set_batch_size(4)
    assert loader.batch_size == loader.n_points


def test_single_batch_loader_sharding_warns_on_remainder(
    sample_data,
    named_sharding,
):
    with pytest.warns(UserWarning, match="Remove the last"):
        loader = SingleBatchJaxLoader(
            data=sample_data(n=5, m=2, seed=11),
            sharding=named_sharding,
            seed=0,
        )
    assert loader._points_per_shard == 4  # noqa: SLF001


def test_single_batch_loader_shard_batch(
    monkeypatch,
    sample_data,
    single_device_sharding,
):
    loader = SingleBatchJaxLoader(
        data=sample_data(n=4, m=2, seed=12),
        sharding=single_device_sharding,
        seed=0,
    )

    calls: list[tuple[jsd.Sharding, tuple[int, int]]] = []

    def fake_make_array(sharding, arr):
        calls.append((sharding, arr.shape))
        return arr

    monkeypatch.setattr(
        dataloader_module.jax,
        "make_array_from_process_local_data",
        fake_make_array,
    )

    indices = np.arange(loader.n_points)
    batch = loader._batch_func(indices, 0)  # noqa: SLF001
    assert set(batch) == {"x", "y"}
    assert calls
    assert calls[0][0] is single_device_sharding
    assert calls[0][1][0] == loader.n_points


def test_single_batch_loader_flatten_helpers(sample_data):
    loader = SingleBatchJaxLoader(
        data=sample_data(n=3, m=2, seed=13),
        sharding=None,
        seed=0,
    )
    loader._points_per_shard = loader.n_points  # noqa: SLF001
    children, aux = loader.flatten()
    assert children["x"].shape[0] == loader.n_points  # noqa: SLF001
    assert aux[0] is None

    keyed_children, keyed_aux = loader.flatten_with_keys()
    assert keyed_children[0].name == "_data"
    assert keyed_aux[2] == loader.n_points

    restored = SingleBatchJaxLoader.unflatten((None, None, aux[2]), (children,))
    assert isinstance(restored, SingleBatchJaxLoader)
    assert restored.batch_size == loader.batch_size
