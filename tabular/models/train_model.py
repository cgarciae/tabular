import json
import typing as tp
from pathlib import Path

import einops
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import typer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from importlib.machinery import SourceFileLoader

import elegy


def main(
    config_path: Path = Path("tabular", "models", "config", "catnet.py"),
    data_path: Path = Path("data/processed"),
    debug: bool = False,
    label: str = "Survived",
    train_size: float = 0.8,
):

    if debug:
        import debugpy

        print("Waiting debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    config = SourceFileLoader(config_path.name, config_path.as_posix()).load_module()
    df_train = pd.read_csv(data_path / "train.csv")
    df_train, df_valid = train_test_split(df_train, train_size=train_size)
    df_test = pd.read_csv(data_path / "test.csv")
    feature_metadata = json.loads((data_path / "metadata.json").read_text())

    X_train = {col: np.asarray(values)[:, None] for col, values in df_train.iteritems()}
    y_train = X_train.pop(label)
    X_train = dict(x=X_train)

    X_valid = {col: np.asarray(values)[:, None] for col, values in df_valid.iteritems()}
    y_valid = X_valid.pop(label)
    X_valid = dict(x=X_valid)

    X_test = {col: np.asarray(values)[:, None] for col, values in df_test.iteritems()}
    X_test = dict(x=X_test)

    # rf = RandomForestClassifier()
    # df_train.drop(columns=["Survived"], inplace=True)
    # df_valid.drop(columns=["Survived"], inplace=True)
    # rf.fit(df_train.values, y_train)
    # print(rf.score(df_valid.values, y_valid))
    # exit()

    model = config.get_model(
        feature_metadata, [label], X_train, y_train, X_valid, y_valid
    )

    model.fit(
        X_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(X_valid, y_valid),
    )


if __name__ == "__main__":
    typer.run(main)
