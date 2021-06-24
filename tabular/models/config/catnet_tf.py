import math
import typing as tp
import einops
import tensorflow as tf


class CatNet(tf.keras.Model):
    def __init__(
        self,
        metadata: tp.Dict[str, tp.Dict[str, tp.Any]],
        labels: tp.List[str],
        embedding_channels: int = 32,
        layer_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.features = {k: v for k, v in metadata.items() if k not in labels}
        self.labels = {k: v for k, v in metadata.items() if k in labels}
        self.embedding_channels = embedding_channels
        self.layer_channels = layer_channels
        self.dropout = dropout

        self.linear_1 = tf.keras.layers.Dense(self.layer_channels)
        self.dropout_1 = tf.keras.layers.Dropout(self.dropout)
        self.linear_out = tf.keras.layers.Dense(1)
        self.embedding_layers = {}

        for feature, feature_metadata in self.features.items():
            kind = feature_metadata["kind"]

            if kind == "continuous":
                self.embedding_layers[feature] = tf.keras.layers.Dense(
                    self.embedding_channels
                )
                # values = tf.keras.layers.Dense(self.embedding_channels)(values)
            elif kind == "categorical":
                vocab_size = feature_metadata["size"]
                self.embedding_layers[feature] = tf.keras.layers.Embedding(
                    vocab_size,
                    math.ceil(vocab_size * 1.75),
                )
            else:
                raise ValueError(f"unknown kind '{kind}'")

    def call(self, x):
        x = x["x"]
        x = self.embed_features(x)

        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = tf.nn.gelu(x)

        logits = self.linear_out(x)

        return logits

    def embed_features(self, x):
        assert set(x.keys()) == set(self.features.keys())

        xs = []
        for feature, feature_metadata in self.features.items():
            kind = feature_metadata["kind"]
            values = x[feature]

            if kind == "continuous":
                pass
                # values = self.embedding_layers[feature](values)
                # values = tf.nn.gelu(values)
            elif kind == "categorical":
                values = values[:, 0]
                values = self.embedding_layers[feature](values)
            else:
                raise ValueError(f"unknown kind '{kind}'")

            xs.append(values)

        x = tf.concat(xs, axis=1)

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

    model = CatNet(
        metadata=feature_metadata,
        labels=labels,
        embedding_channels=32,
        layer_channels=256,
        dropout=0.0,
    )

    model.compile(
        loss=tf.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.metrics.BinaryAccuracy(),
        optimizer=tf.optimizers.Adam(3e-4),
        # run_eagerly=True,
    )

    model(X_train)
    model.summary()

    return model
