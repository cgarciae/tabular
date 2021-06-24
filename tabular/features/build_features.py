import json
from pathlib import Path
import pickle
import typing as tp

import pandas as pd
import numpy as np
import typer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    LabelEncoder,
    MinMaxScaler,
)
from sklearn.impute import SimpleImputer


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

    def fit(self, X: pd.DataFrame, y=None) -> "DateTransformer":
        df = pd.DataFrame(index=X.index)

        for col in X.columns:
            df[f"{col}_year"] = X[col].dt.year
            df[f"{col}_dayofyear"] = X[col].dt.day
            df[f"{col}_monthofyear"] = X[col].dt.month
            df[f"{col}_dayofweek"] = X[col].dt.dayofweek
            df[f"{col}_weekofyear"] = X[col].dt.isocalendar().week

        self.output_features_ = df.columns
        self.encoder_.fit(df)

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:

        df = pd.DataFrame(index=X.index)

        for col in X.columns:
            df[f"{col}_year"] = X[col].dt.year
            df[f"{col}_dayofyear"] = X[col].dt.day
            df[f"{col}_monthofyear"] = X[col].dt.month
            df[f"{col}_dayofweek"] = X[col].dt.dayofweek
            df[f"{col}_weekofyear"] = X[col].dt.isocalendar().week

        df[df.columns] = self.encoder_.transform(df) + 1

        return df


class GenericTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, labels: tp.Sequence[str], exclude: tp.Optional[tp.Set[str]] = None
    ):
        self.labels = labels
        self.exclude = exclude

        self.label_transformer_ = LabelEncoder()
        self.numeric_transformer_ = Pipeline(
            [
                ("scale", MinMaxScaler()),
                ("inpute", SimpleImputer(strategy="median", add_indicator=True)),
            ]
        )
        self.categoric_transformer_ = Pipeline(
            [
                ("inpute", SimpleImputer(strategy="constant", fill_value="_NaN")),
                (
                    "encode",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )

        self.date_transformer_ = DateTransformer()

    def fit(self, X: pd.DataFrame, y=None) -> "GenericTransformer":

        self.numeric_features_ = []
        self.categoric_features_ = []
        self.date_features_ = []

        for col, dtype in X.dtypes.iteritems():

            if col in self.labels:
                continue
            elif self.exclude is not None and col in self.exclude:
                continue
            elif dtype == np.float32 or dtype == np.float64:
                self.numeric_features_.append(col)
            elif dtype == np.int32 or dtype == np.int64 or dtype == np.dtype("object"):
                self.categoric_features_.append(col)
            elif dtype == np.dtype("<M8[ns]"):
                self.date_features_.append(col)

        self.numeric_transformer_.fit(X[self.numeric_features_])
        self.categoric_transformer_.fit(X[self.categoric_features_])
        self.date_transformer_.fit(X[self.date_features_])
        self.label_transformer_.fit(X[self.labels])

        # numeric missing
        indicator = self.numeric_transformer_.steps[1][1].indicator_
        if indicator is not None:
            self.numeric_missing_ = [
                f"{self.numeric_features_[i]}_missing" for i in indicator.features_
            ]
        else:
            self.numeric_missing_ = []

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        df = pd.DataFrame(index=X.index)
        cols = X.columns

        # numerical
        numeric_cols = self.numeric_features_ + self.numeric_missing_
        df[numeric_cols] = self.numeric_transformer_.transform(
            X[self.numeric_features_]
        )
        df[self.numeric_missing_] = df[self.numeric_missing_].astype(int)

        # categorical
        df[self.categoric_features_] = (
            self.categoric_transformer_.transform(X[self.categoric_features_]) + 1
        ).astype(int)

        # date
        result = self.date_transformer_.transform(X[self.date_features_])
        df[result.columns] = result

        # labels
        if all(label in cols for label in self.labels):
            df[self.labels] = self.label_transformer_.transform(X[self.labels])

        return df

    def get_metadata(self) -> tp.Dict[str, tp.Dict[str, tp.Any]]:

        metadata = {}

        metadata.update(
            {col: dict(kind="continuous") for col in self.numeric_features_}
        )
        metadata.update(
            {col: dict(kind="categorical", size=2) for col in self.numeric_missing_}
        )
        metadata.update(
            {
                col: dict(
                    kind="categorical",
                    size=len(categories) + 1,
                )
                for col, categories in zip(
                    self.categoric_features_,
                    self.categoric_transformer_.steps[1][1].categories_,
                )
            }
        )
        if len(self.labels) == 1:
            metadata.update(
                {
                    col: dict(
                        kind="categorical",
                        size=len(self.label_transformer_.classes_),
                    )
                    for col in self.labels
                }
            )
        else:
            metadata.update(
                {
                    col: dict(
                        kind="categorical",
                        size=len(categories),
                    )
                    for col, categories in zip(
                        self.labels,
                        self.label_transformer_.classes_,
                    )
                }
            )
        metadata.update(
            {
                col: dict(
                    kind="categorical",
                    size=len(categories) + 1,
                )
                for col, categories in zip(
                    self.date_transformer_.output_features_,
                    self.date_transformer_.encoder_.categories_,
                )
            }
        )

        return metadata


def main(
    data_path: Path = Path("data/raw/titanic"),
    output_path: Path = Path("data/processed"),
    debug: bool = False,
):

    if debug:
        import debugpy

        print("Waiting debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    train_path = data_path / "train.csv"
    test_path = data_path / "test.csv"

    transformer = GenericTransformer(
        labels=["Survived"],
        exclude={"Name", "PassengerId", "Ticket"},
    )

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train: pd.DataFrame = transformer.fit_transform(df_train)
    df_test = transformer.transform(df_test)
    metadata = transformer.get_metadata()

    output_path.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_path / "train.csv", index=False)
    df_test.to_csv(output_path / "test.csv", index=False)
    (output_path / "transformer.pkl").write_bytes(pickle.dumps(transformer))
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=4))


if __name__ == "__main__":
    typer.run(main)
