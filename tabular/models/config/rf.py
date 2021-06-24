import elegy
import jax
import jax.numpy as jnp
import typing as tp
import einops
import numpy as np
import optax
from sklearn.ensemble import RandomForestClassifier


class RandomForrest:
    def __init__(self, estimator):
        self.estimator = estimator

    def init(self, *args, **kwargs):
        pass

    def summary(self, *args, **kwargs):
        pass

    def fit(self, X, y, validation_data, **kwargs):
        X_valid, y_valid = validation_data

        X = np.concatenate([v for v in X["x"].values()], axis=1)
        X_valid = np.concatenate([v for v in X_valid["x"].values()], axis=1)

        self.estimator.fit(X, y[:, 0])

        print("score", self.estimator.score(X_valid, y_valid[:, 0]))
        return self


batch_size = None
epochs = None


def get_model(
    feature_metadata,
    labels,
    X_train,
    y_train,
    X_valid,
    y_valid,
):

    model = RandomForrest(
        RandomForestClassifier(
            n_estimators=1500,
            min_samples_split=6,
            min_samples_leaf=6,
        ),
    )

    return model
