import elegy
import jax
import jax.numpy as jnp
import typing as tp
import einops
import optax


class MixerBlock(elegy.Module):
    def __init__(self, elements: int, channels: int):
        self.elements = elements
        self.channels = channels
        super().__init__()

    def call(self, x):
        transpose = lambda x: einops.rearrange(x, "... n d -> ... d n")
        normalize = lambda x: elegy.nn.LayerNormalization()(x)
        mpl_1 = lambda x: MLP(self.elements)(x)
        mpl_2 = lambda x: MLP(self.channels)(x)

        x0 = x if x.shape[-2] == self.elements else 0.0
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


class Mixer(elegy.Module):
    def __init__(
        self,
        metadata: tp.Dict[str, tp.Dict[str, tp.Any]],
        labels: tp.List[str],
        embedding_channels: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.features = {k: v for k, v in metadata.items() if k not in labels}
        self.labels = {k: v for k, v in metadata.items() if k in labels}
        self.embedding_channels = embedding_channels
        self.num_layers = num_layers

    def call(self, x):
        x = self.embed_features(x)

        # add CLS token
        token = self.add_parameter(
            "token",
            lambda: elegy.initializers.TruncatedNormal()(
                [1, x.shape[-1]],
                jnp.float32,
            ),
        )
        token = einops.repeat(token, "... -> batch ...", batch=x.shape[0])
        x = jnp.concatenate([token, x], axis=1)

        for i in range(self.num_layers):
            x = MixerBlock(
                elements=x.shape[1],
                channels=self.embedding_channels,
            )(x)

        # reduce channels
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
        module=Mixer(
            metadata=feature_metadata,
            labels=labels,
            embedding_channels=96,
            num_layers=2,
        ),
        loss=[
            elegy.losses.BinaryCrossentropy(from_logits=True),
            # elegy.regularizers.GlobalL2(0.0001),
        ],
        metrics=elegy.metrics.BinaryAccuracy(),
        optimizer=optax.adamw(3e-5),
    )

    model.init(
        jax.tree_map(lambda x: x[:batch_size], X_train),
        jax.tree_map(lambda x: x[:batch_size], y_train),
    )
    model.summary(jax.tree_map(lambda x: x[:batch_size], X_train), depth=1)

    return model
