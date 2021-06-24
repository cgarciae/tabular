import math
import elegy
import jax
import jax.numpy as jnp
import typing as tp
import einops
import optax


class Embedding(elegy.Module):
    def __init__(self, vocab_size: int, output_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size

    def call(self, x):
        x = jax.nn.one_hot(x, self.vocab_size)
        x = elegy.nn.Linear(self.output_size, with_bias=False)(x)
        return x


class CatNet(elegy.Module):
    def __init__(
        self,
        metadata: tp.Dict[str, tp.Dict[str, tp.Any]],
        labels: tp.List[str],
        layer_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.features = {k: v for k, v in metadata.items() if k not in labels}
        self.labels = {k: v for k, v in metadata.items() if k in labels}
        self.layer_channels = layer_channels
        self.dropout = dropout

    def call(self, x):
        x = self.embed_features(x)
        self.add_summary("embed_features", self.embed_features, x)

        x = elegy.nn.Linear(self.layer_channels)(x)
        x = elegy.nn.Dropout(self.dropout)(x)
        x = jax.nn.gelu(x)

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
                values = values[:, 0]
                vocab_size = feature_metadata["size"]
                # values = Embedding(
                values = elegy.nn.Embedding(
                    vocab_size + 10,
                    math.ceil(vocab_size * 2.0),
                )(values)
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
        module=CatNet(
            metadata=feature_metadata,
            labels=labels,
            layer_channels=64,
            dropout=0.0,
        ),
        loss=[
            elegy.losses.BinaryCrossentropy(from_logits=True),
            # elegy.regularizers.GlobalL2(0.0001),
        ],
        metrics=elegy.metrics.BinaryAccuracy(),
        optimizer=optax.adamw(3e-4),
        # run_eagerly=True,
    )

    model.init(X_train, y_train)
    model.summary(X_train)

    return model
