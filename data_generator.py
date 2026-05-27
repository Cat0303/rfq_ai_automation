"""
Synthetic RFQ data generator for the AI-Powered RFQ Automation project.

All data created by this module is synthetic and intended for educational,
portfolio, and demonstration use only. It does not represent any real company,
customer, transaction, or confidential business record.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
RANDOM_SEED = 42


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _build_parts_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["P-1001", "Industrial temperature sensor", "Sensors", 185.00, 112.00, 180, 7, 0.22],
            ["P-1002", "Hydraulic pump assembly", "Hydraulics", 1420.00, 940.00, 34, 21, 0.64],
            ["P-1003", "PLC control module", "Controls", 890.00, 575.00, 52, 14, 0.31],
            ["P-1004", "Precision ball valve", "Valves", 245.00, 151.00, 120, 10, 0.27],
            ["P-1005", "Servo motor kit", "Motion", 1125.00, 760.00, 40, 18, 0.45],
            ["P-1006", "Food-grade conveyor belt", "Conveyance", 680.00, 438.00, 26, 16, 0.58],
            ["P-1007", "Robotic end effector", "Automation", 2310.00, 1540.00, 15, 28, 0.73],
            ["P-1008", "Industrial safety relay", "Controls", 135.00, 82.00, 240, 6, 0.18],
            ["P-1009", "Stainless pressure fitting", "Fittings", 48.00, 29.00, 650, 5, 0.15],
            ["P-1010", "Vision inspection camera", "Automation", 1760.00, 1185.00, 22, 24, 0.69],
            ["P-1011", "Variable frequency drive", "Controls", 740.00, 482.00, 48, 12, 0.36],
            ["P-1012", "Pneumatic cylinder pack", "Pneumatics", 315.00, 198.00, 95, 9, 0.25],
        ],
        columns=[
            "part_number",
            "description",
            "part_category",
            "base_price",
            "standard_cost",
            "inventory_level",
            "standard_lead_time_days",
            "supplier_risk_score",
        ],
    )


def _build_current_rfqs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["RFQ-2026-001", "Apex Fabrication Group", "Enterprise", "Industrial Manufacturing", "Northeast", "P-1003", 65, "High", 10, "Net 30", "Strategic"],
            ["RFQ-2026-002", "Brightline Medical Systems", "Mid-Market", "Healthcare Equipment", "Midwest", "P-1008", 180, "Medium", 7, "Net 45", "Preferred"],
            ["RFQ-2026-003", "Cedar Ridge Foods", "Enterprise", "Food Processing", "Southeast", "P-1006", 42, "High", 12, "Net 30", "Strategic"],
            ["RFQ-2026-004", "Northstar Energy Services", "Enterprise", "Energy", "Southwest", "P-1002", 18, "Low", 28, "Net 60", "Standard"],
            ["RFQ-2026-005", "Summit Robotics Lab", "Growth", "Automation", "West", "P-1007", 9, "High", 20, "Net 30", "Preferred"],
            ["RFQ-2026-006", "Harbor Logistics Partners", "Mid-Market", "Transportation", "Northeast", "P-1011", 76, "Medium", 14, "Net 45", "Standard"],
            ["RFQ-2026-007", "Keystone Waterworks", "Public Sector", "Utilities", "Mid-Atlantic", "P-1004", 130, "Medium", 11, "Net 60", "Preferred"],
            ["RFQ-2026-008", "Evergreen Packaging Co.", "Enterprise", "Packaging", "West", "P-1009", 420, "Low", 8, "Net 30", "Strategic"],
            ["RFQ-2026-009", "Pioneer Aerospace Components", "Enterprise", "Aerospace", "Southwest", "P-1010", 14, "High", 18, "Net 45", "Strategic"],
            ["RFQ-2026-010", "Metro Transit Maintenance", "Public Sector", "Transportation", "Midwest", "P-1012", 105, "Medium", 10, "Net 60", "Standard"],
            ["RFQ-2026-011", "Clearwater Pharma Supply", "Mid-Market", "Pharmaceuticals", "Northeast", "P-1001", 260, "High", 5, "Net 30", "Preferred"],
            ["RFQ-2026-012", "IronPeak Mining Solutions", "Enterprise", "Mining", "West", "P-1005", 30, "Low", 24, "Net 45", "Standard"],
            ["RFQ-2026-013", "Nova Grid Infrastructure", "Growth", "Utilities", "Southeast", "P-1002", 28, "High", 16, "Net 30", "Preferred"],
            ["RFQ-2026-014", "Lakeside Specialty Foods", "Mid-Market", "Food Processing", "Midwest", "P-1006", 70, "Medium", 15, "Net 45", "Standard"],
        ],
        columns=[
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
        ],
    )


def _build_historical_rfqs(parts_catalog: pd.DataFrame, rows: int = 650) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    customer_segments = ["Enterprise", "Mid-Market", "Growth", "Public Sector"]
    industries = [
        "Industrial Manufacturing",
        "Healthcare Equipment",
        "Food Processing",
        "Energy",
        "Automation",
        "Transportation",
        "Utilities",
        "Packaging",
        "Aerospace",
        "Pharmaceuticals",
        "Mining",
    ]
    urgencies = ["Low", "Medium", "High"]
    customer_tiers = ["Standard", "Preferred", "Strategic"]
    part_categories = parts_catalog["part_category"].unique()

    records = []
    for _ in range(rows):
        segment = rng.choice(customer_segments, p=[0.35, 0.34, 0.18, 0.13])
        industry = rng.choice(industries)
        category = rng.choice(part_categories)
        urgency = rng.choice(urgencies, p=[0.25, 0.46, 0.29])
        tier = rng.choice(customer_tiers, p=[0.44, 0.36, 0.20])
        order_size = int(np.clip(rng.lognormal(mean=4.25, sigma=0.75), 5, 650))

        category_risk = float(
            parts_catalog.loc[parts_catalog["part_category"] == category, "supplier_risk_score"].mean()
        )
        supplier_risk = float(np.clip(rng.normal(category_risk, 0.12), 0.05, 0.95))

        urgency_days = {"Low": 24, "Medium": 15, "High": 8}[urgency]
        lead_time_days = int(np.clip(rng.normal(urgency_days, 5), 3, 45))

        tier_margin_shift = {"Standard": 0.03, "Preferred": 0.00, "Strategic": -0.02}[tier]
        urgency_margin_shift = {"Low": 0.02, "Medium": 0.01, "High": 0.04}[urgency]
        gross_margin_pct = float(
            np.clip(rng.normal(0.34 + tier_margin_shift + urgency_margin_shift, 0.055), 0.16, 0.55)
        )

        price_competitiveness = float(
            np.clip(rng.normal(0.92 - gross_margin_pct * 0.22 + (0.03 if tier == "Strategic" else 0), 0.08), 0.58, 1.12)
        )

        segment_effect = {"Enterprise": 0.95, "Mid-Market": 0.35, "Growth": -0.05, "Public Sector": -0.35}[segment]
        tier_effect = {"Strategic": 1.05, "Preferred": 0.45, "Standard": -0.25}[tier]
        urgency_effect = {"High": 0.35, "Medium": 0.05, "Low": -0.20}[urgency]
        margin_effect = (0.37 - gross_margin_pct) * 5.5
        price_effect = (price_competitiveness - 0.82) * 7.5
        risk_effect = -2.8 * supplier_risk
        lead_time_effect = -0.075 * max(0, lead_time_days - urgency_days)
        size_effect = 0.55 if order_size > 150 else -0.10

        logit = -0.25 + segment_effect + tier_effect + urgency_effect + margin_effect + price_effect + risk_effect + lead_time_effect + size_effect
        win_probability = 1 / (1 + np.exp(-logit))
        won = int(rng.random() < win_probability)

        records.append(
            [
                segment,
                industry,
                category,
                order_size,
                lead_time_days,
                urgency,
                tier,
                round(price_competitiveness, 3),
                round(supplier_risk, 3),
                round(gross_margin_pct, 3),
                won,
            ]
        )

    return pd.DataFrame(
        records,
        columns=[
            "customer_segment",
            "industry",
            "part_category",
            "order_size",
            "lead_time_days",
            "urgency",
            "customer_tier",
            "price_competitiveness",
            "supplier_risk_score",
            "gross_margin_pct",
            "won",
        ],
    )


def generate_sample_data(overwrite: bool = False) -> dict[str, Path]:
    """Generate reproducible synthetic CSV files when they do not already exist."""
    _ensure_data_dir()

    paths = {
        "rfqs": DATA_DIR / "rfqs.csv",
        "parts_catalog": DATA_DIR / "parts_catalog.csv",
        "historical_rfqs": DATA_DIR / "historical_rfqs.csv",
    }

    parts_catalog = _build_parts_catalog()
    current_rfqs = _build_current_rfqs()
    historical_rfqs = _build_historical_rfqs(parts_catalog)

    if overwrite or not paths["parts_catalog"].exists():
        parts_catalog.to_csv(paths["parts_catalog"], index=False)
    if overwrite or not paths["rfqs"].exists():
        current_rfqs.to_csv(paths["rfqs"], index=False)
    if overwrite or not paths["historical_rfqs"].exists():
        historical_rfqs.to_csv(paths["historical_rfqs"], index=False)

    return paths


if __name__ == "__main__":
    created_paths = generate_sample_data()
    for name, path in created_paths.items():
        print(f"{name}: {path}")
