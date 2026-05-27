"""Model utilities for RFQ win probability scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORICAL_FEATURES = [
    "customer_segment",
    "industry",
    "part_category",
    "urgency",
    "customer_tier",
]

NUMERIC_FEATURES = [
    "order_size",
    "lead_time_days",
    "price_competitiveness",
    "supplier_risk_score",
    "gross_margin_pct",
]

TARGET = "won"


@dataclass
class ModelTrainingResult:
    model: Pipeline
    accuracy: float
    roc_auc: float | None
    feature_names: list[str]
    training_sample_size: int
    test_sample_size: int


def build_model_pipeline() -> Pipeline:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ]
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def train_win_probability_model(historical_rfqs: pd.DataFrame) -> ModelTrainingResult:
    missing_columns = [col for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET] if col not in historical_rfqs.columns]
    if missing_columns:
        raise ValueError(f"Historical RFQ data is missing required model columns: {missing_columns}")

    model_data = historical_rfqs[CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET]].dropna()
    if model_data[TARGET].nunique() < 2:
        raise ValueError("Historical RFQ data must contain both won and lost examples to train the model.")

    X = model_data[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = model_data[TARGET].astype(int)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))

    roc_auc = None
    if y_test.nunique() == 2 and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, probabilities))

    feature_names = _extract_feature_names(model)

    return ModelTrainingResult(
        model=model,
        accuracy=accuracy,
        roc_auc=roc_auc,
        feature_names=feature_names,
        training_sample_size=len(X_train),
        test_sample_size=len(X_test),
    )


def _extract_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return CATEGORICAL_FEATURES + NUMERIC_FEATURES


def predict_win_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.predict(features).astype(float)
