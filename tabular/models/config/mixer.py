import elegy
import jax
import jax.numpy as jnp
import typing as tp
import einops
import optax


class MixerBlock(elegy.Module):
    def __init__(self, length: int, channels: int):
        self.length = length
        self.channels = channels
        super().__init__()

    def call(self, x):
        transpose = lambda x: einops.rearrange(x, "... n d -> ... d n")
        normalize = lambda x: elegy.nn.LayerNormalization()(x)
        mpl_1 = lambda x: MLP(self.length)(x)
        mpl_2 = lambda x: MLP(self.channels)(x)

        x0 = x if x.shape[-2] == self.length else 0.0
        x = x0 + transpose(mpl_1(transpose(normalize(x))))

        x0 = x if x.shape[-1] == self.channels else 0.0
        x = x0 + normalize(mpl_2(x))

        return x


class MLP(elegy.Module):
    def __init__(self, units: int):
        self.units = units
        super().__init__()

    def call(self, x):

        x = elegy.nn.Linear(self.units)(x)
        x = jax.nn.gelu(x)
        x = elegy.nn.Linear(self.units)(x)

        return x


class Transformer(elegy.Module):
    def __init__(
        self,
        metadata: tp.Dict[str, tp.Dict[str, tp.Any]],
        labels: tp.List[str],
        embedding_channels: int = 32,
        dropout: float = 0.0,
        num_layers: int = 2,
    ):
        super().__init__()
        self.features = {k: v for k, v in metadata.items() if k not in labels}
        self.labels = {k: v for k, v in metadata.items() if k in labels}
        self.embedding_channels = embedding_channels
        self.dropout = dropout
        self.num_layers = num_layers

    def call(self, x):
        x = self.embed_features(x)

        # add CLS token
        zeros = jnp.zeros([x.shape[0], 1, x.shape[2]])
        x = jnp.concatenate([zeros, x], axis=1)

        # add positional embeddings
        positional_embeddings = self.add_parameter(
            "positional_embeddings",
            lambda: elegy.initializers.TruncatedNormal()(
                x.shape[1:],
                jnp.float32,
            ),
        )
        positional_embeddings = einops.repeat(
            positional_embeddings, "... -> batch ...", batch=x.shape[0]
        )
        x += positional_embeddings

        x = elegy.nn.TransformerEncoder(
            lambda: elegy.nn.TransformerEncoderLayer(
                head_size=self.embedding_channels,
                output_size=self.embedding_channels,
                num_heads=8,
            ),
            num_layers=self.num_layers,
        )(x)

        x = x[:, 0]
        logits = elegy.nn.Linear(1)(x)

        return logits

    def embed_features(self, x):
        assert set(x.keys()) == set(self.features.keys())

        xs = []
        for feature, feature_metadata in self.features.items():
            kind = feature_metadata["kind"]
            values = x[feature]

            if kind == "continuous":
                values = elegy.nn.Linear(self.embedding_channels)(values)
                values = jax.nn.relu(values)
                values = elegy.nn.Linear(self.embedding_channels)(values)
            elif kind == "categorical":
                values = values[:, 0]
                vocab_size = feature_metadata["size"]
                values = elegy.nn.Embedding(
                    vocab_size=vocab_size,
                    embed_dim=self.embedding_channels,
                )(values)
            else:
                raise ValueError(f"unknown kind '{kind}'")

            xs.append(values)

        x = jnp.stack(xs, axis=1)

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
        module=Transformer(
            metadata=feature_metadata,
            labels=labels,
            embedding_channels=96,
            dropout=0.0,
            num_layers=2,
        ),
        loss=[
            elegy.losses.BinaryCrossentropy(from_logits=True),
            # elegy.regularizers.GlobalL2(0.005),
        ],
        metrics=elegy.metrics.BinaryAccuracy(),
        optimizer=optax.adamw(1e-4),
    )

    model.init(X_train, y_train)
    model.summary(X_train)

    return model
