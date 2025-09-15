import numpy as np

from trainax import JaxLoader, SingleBatchJaxLoader


def get_test_data(
    seed: int,
    n: int,
    m: int,
) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)

    xs = rng.normal(size=(n, m))
    ys = np.sin(xs)

    return {"x": xs, "y": ys}


class TestLoader:
    seed = 42
    n = 100
    m = 10
    data = get_test_data(seed, n, m)

    def test_init(self):
        loader = JaxLoader(
            data=self.data,
            batch_size=50,
            sharding=None,
            seed=self.seed,
        )
        assert loader.n_batches == 2
        assert len(loader) == 2
        assert loader.n_points == self.n
        assert loader.batch_size == self.n // 2

    def test_single_batch_init(self):
        loader = SingleBatchJaxLoader(
            data=self.data,
            sharding=None,
            seed=self.seed,
        )
        assert loader.n_batches == 1
        assert len(loader) == 1
        assert loader.n_points == self.n
        assert loader.batch_size == self.n

    # TODO: test getter
    # TODO: test iterator
    # TODO: test sharding
    # TODO: test warnings
