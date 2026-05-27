"""Reusable RFQ automation pipeline functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_generator import generate_sample_data
from email_generator import add_emails_to_quotes
from model_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ModelTrainingResult,
    predict_win_probability,
    train_win_probability_model,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

RFQ_REQUIRED_COLUMNS = [
    "rfq_id",
    "customer_name",
    "customer_segment",
    "industry",
    "region",
    "part_number",
    "quantity",
    "urgency",
    "requested_lead_time_days",
    "payment_terms",
    "customer_tier",
]

CATALOG_REQUIRED_COLUMNS = [
    "part_number",
    "description",
    "part_category",
    "base_price",
    "standard_cost",
    "inventory_level",
    "standard_lead_time_days",
    "supplier_risk_score",
]

HISTORICAL_REQUIRED_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["won"]


def _validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")


def _csv_path(file_name: str) -> Path:
    """Resolve CSV files from data/ first, with root-level fallback for older repos."""
    data_path = DATA_DIR / file_name
    root_path = PROJECT_DIR / file_name
    if data_path.exists():
        return data_path
    if root_path.exists():
        return root_path
    return data_path


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all CSV inputs and validate required columns."""
    generate_sample_data(overwrite=False)

    rfqs = pd.read_csv(_csv_path("rfqs.csv"))
    parts_catalog = pd.read_csv(_csv_path("parts_catalog.csv"))
    historical_rfqs = pd.read_csv(_csv_path("historical_rfqs.csv"))

    _validate_columns(rfqs, RFQ_REQUIRED_COLUMNS, "rfqs.csv")
    _validate_columns(parts_catalog, CATALOG_REQUIRED_COLUMNS, "parts_catalog.csv")
    _validate_columns(historical_rfqs, HISTORICAL_REQUIRED_COLUMNS, "historical_rfqs.csv")

    return rfqs, parts_catalog, historical_rfqs


def merge_rfqs_with_catalog(rfqs: pd.DataFrame, parts_catalog: pd.DataFrame) -> pd.DataFrame:
    """Join current RFQs to the part catalog and flag unmatched part numbers."""
    merged = rfqs.merge(parts_catalog, on="part_number", how="left", indicator=True)
    merged["matched_part"] = merged["_merge"].eq("both")
    merged["unmatched_part"] = ~merged["matched_part"]
    return merged.drop(columns=["_merge"])


def calculate_quote_price(merged_rfqs: pd.DataFrame) -> pd.DataFrame:
    """Apply business pricing rules and calculate quote values and margin."""
    quotes = merged_rfqs.copy()

    for column in ["base_price", "standard_cost", "inventory_level", "supplier_risk_score", "standard_lead_time_days"]:
        quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    quotes["quantity"] = pd.to_numeric(quotes["quantity"], errors="coerce").fillna(0).astype(int)
    quotes["requested_lead_time_days"] = pd.to_numeric(quotes["requested_lead_time_days"], errors="coerce")

    unit_prices = []
    pricing_notes = []

    for _, row in quotes.iterrows():
        if not bool(row.get("matched_part", False)) or pd.isna(row.get("base_price")):
            unit_prices.append(np.nan)
            pricing_notes.append("Unmatched part number - manual pricing review required")
            continue

        price = float(row["base_price"])
        notes = [f"Base catalog price ${price:,.2f}"]

        urgency = str(row.get("urgency", "")).strip().lower()
        if urgency == "high":
            price *= 1.10
            notes.append("High urgency surcharge +10%")
        elif urgency == "medium":
            price *= 1.05
            notes.append("Medium urgency surcharge +5%")

        quantity = int(row.get("quantity", 0))
        if quantity > 250:
            price *= 0.92
            notes.append("Volume discount -8%")
        elif quantity > 100:
            price *= 0.95
            notes.append("Volume discount -5%")

        customer_tier = str(row.get("customer_tier", "")).strip().lower()
        if customer_tier == "strategic":
            price *= 0.97
            notes.append("Strategic relationship discount -3%")

        supplier_risk_score = float(row.get("supplier_risk_score", 0) or 0)
        if supplier_risk_score > 0.70:
            price *= 1.04
            notes.append("Supplier risk surcharge +4%")

        inventory_level = float(row.get("inventory_level", 0) or 0)
        if inventory_level < quantity:
            price *= 1.06
            notes.append("Inventory constraint surcharge +6%")

        unit_prices.append(round(price, 2))
        pricing_notes.append("; ".join(notes))

    quotes["unit_quote_price"] = unit_prices
    quotes["total_quote_value"] = (quotes["unit_quote_price"] * quotes["quantity"]).round(2)
    quotes["estimated_gross_margin_pct"] = (
        (quotes["unit_quote_price"] - quotes["standard_cost"]) / quotes["unit_quote_price"]
    ).replace([np.inf, -np.inf], np.nan)
    quotes["estimated_gross_margin_pct"] = quotes["estimated_gross_margin_pct"].round(4)
    quotes["pricing_notes"] = pricing_notes

    quotes["price_competitiveness"] = (quotes["base_price"] / quotes["unit_quote_price"]).replace([np.inf, -np.inf], np.nan)
    quotes["price_competitiveness"] = quotes["price_competitiveness"].fillna(0.75).round(3)
    quotes["lead_time_days"] = quotes["requested_lead_time_days"].fillna(quotes["standard_lead_time_days"]).fillna(14).astype(int)
    quotes["order_size"] = quotes["quantity"]
    quotes["gross_margin_pct"] = quotes["estimated_gross_margin_pct"].fillna(0.25)

    return quotes


def prepare_model_features(historical_rfqs: pd.DataFrame) -> ModelTrainingResult:
    """Train a scikit-learn pipeline and return model metrics."""
    return train_win_probability_model(historical_rfqs)


def score_current_rfqs(priced_rfqs: pd.DataFrame, model_result: ModelTrainingResult) -> pd.DataFrame:
    """Predict win probability and assign sales prioritization guidance."""
    scored = priced_rfqs.copy()
    model_features = scored[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    model_features = model_features.fillna(
        {
            "customer_segment": "Unknown",
            "industry": "Unknown",
            "part_category": "Unknown",
            "urgency": "Medium",
            "customer_tier": "Standard",
            "order_size": 1,
            "lead_time_days": 14,
            "price_competitiveness": 0.80,
            "supplier_risk_score": 0.50,
            "gross_margin_pct": 0.25,
        }
    )

    scored["win_probability"] = predict_win_probability(model_result.model, model_features).round(4)
    scored.loc[scored["unmatched_part"], "win_probability"] = np.nan

    conditions = [
        scored["win_probability"].ge(0.70),
        scored["win_probability"].between(0.40, 0.6999),
        scored["win_probability"].lt(0.40),
    ]
    priority_values = ["High Priority", "Sales Review", "Strategic Review"]
    scored["priority_status"] = np.select(conditions, priority_values, default="Manual Review")

    action_map = {
        "High Priority": "Fast-track quote and follow up within 24 hours",
        "Sales Review": "Review pricing and delivery terms",
        "Strategic Review": "Escalate to manager or adjust offer strategy",
        "Manual Review": "Validate catalog match and complete manual review",
    }
    scored["recommended_action"] = scored["priority_status"].map(action_map)

    return scored


def save_final_quotes(final_quotes: pd.DataFrame) -> Path:
    """Save final quote output to outputs/final_quotes.csv."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "final_quotes.csv"
    final_quotes.to_csv(output_path, index=False)
    return output_path


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ModelTrainingResult, Path]:
    """Run the full RFQ workflow from input data to final quote output."""
    rfqs, parts_catalog, historical_rfqs = load_data()
    merged = merge_rfqs_with_catalog(rfqs, parts_catalog)
    priced = calculate_quote_price(merged)
    model_result = prepare_model_features(historical_rfqs)
    scored = score_current_rfqs(priced, model_result)
    final_quotes = add_emails_to_quotes(scored, use_openai=False)
    output_path = save_final_quotes(final_quotes)
    return rfqs, parts_catalog, historical_rfqs, final_quotes, model_result, output_path


if __name__ == "__main__":
    _, _, _, final_quotes, metrics, saved_path = run_pipeline()
    print(f"Saved {len(final_quotes)} quotes to {saved_path}")
    print(f"Model accuracy: {metrics.accuracy:.3f}")
    if metrics.roc_auc is not None:
        print(f"Model ROC-AUC: {metrics.roc_auc:.3f}")
