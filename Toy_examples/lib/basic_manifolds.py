import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve, make_circles, make_moons


class BaseManifold:
    def __init__(self, noise: float = 0.0, a: float = 0.2) -> None:
        self.noise = noise
        self.a = a
        self.data_dim = 2
        self.x_max, self.x_min = None, None
        self.y_max, self.y_min = None, None

    def _generate_data(self, n_samples):
        raise NotImplementedError("Subclasses must implement this method")

    def _preprocess_data(self, data):
        return data

    def _normalize_data(self, data):
        data[:, 0] = (
            2 * self.a * (data[:, 0] - self.x_min) / (self.x_max - self.x_min) - self.a
        )
        data[:, 1] = (
            2 * self.a * (data[:, 1] - self.y_min) / (self.y_max - self.y_min) - self.a
        )
        return data

    def sample(self, n):
        data = self._generate_data(n)
        data = self._preprocess_data(data)
        data = self._normalize_data(data)
        return data

    def data_iter(self, batch_size: int = 32, maxiter: int = 1000):
        for _ in range(maxiter):
            yield self.sample(batch_size)


class SwissRoll(BaseManifold):
    def _generate_data(self, n_samples):
        data, _ = make_swiss_roll(n_samples=n_samples, noise=self.noise)
        return data

    def _preprocess_data(self, data):
        return np.stack([data[:, 0], data[:, 2]], axis=1)

    def __init__(self, noise: float = 0.0, a: float = 0.2) -> None:
        super().__init__(noise, a)
        data = self._preprocess_data(self._generate_data(10000))
        self.x_max, self.x_min = data[:, 0].max(), data[:, 0].min()
        self.y_max, self.y_min = data[:, 1].max(), data[:, 1].min()


class SCurve(BaseManifold):
    def _generate_data(self, n_samples):
        data, _ = make_s_curve(n_samples=n_samples, noise=self.noise)
        return data

    def _preprocess_data(self, data):
        return np.stack([data[:, 0], data[:, 2]], axis=1)

    def __init__(self, noise: float = 0.0, a: float = 0.2) -> None:
        super().__init__(noise, a)
        data = self._preprocess_data(self._generate_data(10000))
        self.x_max, self.x_min = data[:, 0].max(), data[:, 0].min()
        self.y_max, self.y_min = data[:, 1].max(), data[:, 1].min()


class Circles(BaseManifold):
    def _generate_data(self, n_samples):
        data, _ = make_circles(n_samples=n_samples, noise=self.noise)
        return data

    def __init__(self, noise: float = 0.0, a: float = 0.2) -> None:
        super().__init__(noise, a)
        data = self._generate_data(10000)
        self.x_max, self.x_min = data[:, 0].max(), data[:, 0].min()
        self.y_max, self.y_min = data[:, 1].max(), data[:, 1].min()


class Moons(BaseManifold):
    def _generate_data(self, n_samples):
        data, _ = make_moons(n_samples=n_samples, noise=self.noise)
        return data

    def __init__(self, noise: float = 0.0, a: float = 0.2) -> None:
        super().__init__(noise, a)
        data = self._generate_data(10000)
        self.x_max, self.x_min = data[:, 0].max(), data[:, 0].min()
        self.y_max, self.y_min = data[:, 1].max(), data[:, 1].min()
