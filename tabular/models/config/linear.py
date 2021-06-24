import elegy
import jax
import jax.numpy as jnp
import typing as tp
import einops
import optax


class LinearClassifier(elegy.Module):
    def __init__(
        self,
        metadata: tp.Dict[str, tp.Dict[str, tp.Any]],
        labels: tp.List[str],
    ):
        super().__init__()
        self.features = {k: v for k, v in metadata.items() if k not in labels}
        self.labels = {k: v for k, v in metadata.items() if k in labels}

    def call(self, x):
        x = self.embed_features(x)

        logits = elegy.nn.Linear(1)(x)

        return logits

    def embed_features(self, x):
        assert set(x.keys()) == set(self.features.keys())

        xs = []
        for feature, feature_metadata in self.features.items():
            kind = feature_metadata["kind"]
            values = x[feature]

            if kind == "continuous":
                pass
            elif kind == "categorical":
                vocab_size = feature_metadata["size"]
                values = values[:, 0]
                values = jax.nn.one_hot(values, vocab_size)
            else:
                raise ValueError(f"unknown kind '{kind}'")

            xs.append(values)

        x = jnp.concatenate(xs, axis=1)

        return x


batch_size: int = 16
epochs: int = 1000


def get_model(
    feature_metadata,
    labels,
    X_train,
    y_train,
    X_valid,
    y_valid,
):

    model = elegy.Model(
        module=LinearClassifier(
            metadata=feature_metadata,
            labels=labels,
        ),
        loss=[
            elegy.losses.BinaryCrossentropy(from_logits=True),
            elegy.regularizers.GlobalL2(0.0001),
        ],
        metrics=elegy.metrics.BinaryAccuracy(),
        optimizer=optax.adamw(3e-3),
    )

    model.init(X_train, y_train)
    model.summary(X_train)

    return model
