from pathlib import Path

import pandas as pd
import typer


def main(
    data_path: Path = Path("data/processed"),
    debug: bool = False,
):

    if debug:
        import debugpy

        print("Waiting debugger...")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    df_train = pd.read_csv(data_path / "train.csv")
    df_test = pd.read_csv(data_path / "test.csv")


if __name__ == "__main__":
    typer.run(main)
