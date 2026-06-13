from pathlib import Path

import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


TRAIN_FILE = Path("train_reviews.csv")
TEST_FILE = Path("test_reviews.csv")
TEXT_COLUMN = "text_"
LABEL_COLUMN = "label"
REAL_LABEL = "OR"
FAKE_LABEL = "CG"


def _load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run prepare_data.py before starting the app."
        )

    df = pd.read_csv(path)
    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    return df[df[TEXT_COLUMN] != ""]


def train_model() -> tuple[Pipeline, dict[str, object]]:
    train_df = _load_reviews(TRAIN_FILE)
    test_df = _load_reviews(TEST_FILE)

    pipeline = Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                ngram_range=(1, 2),
                                max_features=70000,
                                min_df=2,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                max_features=40000,
                                min_df=2,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0),
            ),
        ]
    )

    pipeline.fit(train_df[TEXT_COLUMN], train_df[LABEL_COLUMN])

    predictions = pipeline.predict(test_df[TEXT_COLUMN])
    metrics = {
        "accuracy": accuracy_score(test_df[LABEL_COLUMN], predictions),
        "report": classification_report(
            test_df[LABEL_COLUMN],
            predictions,
            labels=[REAL_LABEL, FAKE_LABEL],
            output_dict=True,
            zero_division=0,
        ),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }

    return pipeline, metrics


def predict_review(model: Pipeline, review_text: str) -> dict[str, object]:
    text = review_text.strip()
    if not text:
        raise ValueError("Review text cannot be empty.")

    probabilities = model.predict_proba([text])[0]
    class_probabilities = dict(zip(model.classes_, probabilities, strict=True))
    real_probability = float(class_probabilities.get(REAL_LABEL, 0.0))
    fake_probability = float(class_probabilities.get(FAKE_LABEL, 0.0))
    predicted_label = REAL_LABEL if real_probability >= fake_probability else FAKE_LABEL

    return {
        "label": predicted_label,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
    }
