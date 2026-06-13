from pathlib import Path

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError as exc:
    missing_package = exc.name
    raise SystemExit(
        f"Missing Python package: {missing_package}\n"
        "Run this script with the project virtual environment:\n"
        r"  .\venv\Scripts\python.exe prepare_data.py"
    ) from exc


DATA_FILE = Path("fake_reviews_dataset.csv")
TEXT_COLUMN = "text_"
LABEL_COLUMN = "label"
REAL_LABEL = "OR"
FAKE_LABEL = "CG"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    print("Starting data preparation...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Put the CSV in this folder or update DATA_FILE."
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""]

    valid_labels = {REAL_LABEL, FAKE_LABEL}
    bad_labels = sorted(set(df[LABEL_COLUMN]) - valid_labels)
    if bad_labels:
        raise ValueError(
            f"Unexpected labels: {bad_labels}. Expected only {sorted(valid_labels)}."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COLUMN],
    )

    train_df.to_csv("train_reviews.csv", index=False)
    test_df.to_csv("test_reviews.csv", index=False)

    label_counts = df[LABEL_COLUMN].value_counts()

    print(f"Total rows loaded: {len(df)}")
    print(f"Real reviews ({REAL_LABEL}): {label_counts.get(REAL_LABEL, 0)}")
    print(f"Fake reviews ({FAKE_LABEL}): {label_counts.get(FAKE_LABEL, 0)}")
    print(f"Training set: {len(train_df)} reviews")
    print(f"Testing set:  {len(test_df)} reviews")
    print("Data preparation COMPLETE!")


if __name__ == "__main__":
    main()
