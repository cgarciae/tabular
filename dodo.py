def task_make_dataset():
    cmd = ["bash", "tabular/data/make_dataset.sh"]
    return dict(
        actions=[cmd],
        uptodate=[True],
        file_dep=[],
        targets=[
            "data/raw/titanic/gender_submission.csv",
            "data/raw/titanic/train.csv",
            "data/raw/titanic/test.csv",
        ],
        verbosity=2,
    )
