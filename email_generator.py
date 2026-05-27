"""Quote email generation with optional OpenAI support and a reliable fallback."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "TBD"


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "TBD"


def generate_template_email(row: pd.Series | dict[str, Any]) -> str:
    """Create a professional quote email without requiring an AI API key."""
    customer_name = row.get("customer_name", "Customer")
    description = row.get("description", "requested part")
    quantity = row.get("quantity", "the requested quantity")
    unit_price = _money(row.get("unit_quote_price"))
    total_value = _money(row.get("total_quote_value"))
    lead_time = row.get("lead_time_days", row.get("requested_lead_time_days", "TBD"))
    rfq_id = row.get("rfq_id", "your RFQ")

    return f"""Subject: Quote for {rfq_id} - {description}

Hello {customer_name} Team,

Thank you for the opportunity to quote your request for {quantity} units of {description}. Based on the requested quantity, delivery timing, and current catalog availability, we are pleased to provide the following quote:

- Unit quote price: {unit_price}
- Total quote value: {total_value}
- Estimated lead time: {lead_time} business days

Please let us know if you would like us to review alternate delivery options, volume pricing, or updated payment terms. We would be glad to support the next step in your purchasing process.

Best regards,

Karishma Shaik
Sales Operations Analytics"""


def _get_openai_api_key() -> str | None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    try:
        import streamlit as st

        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def _openai_email(row: pd.Series | dict[str, Any]) -> str | None:
    api_key = _get_openai_api_key()
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = f"""
Write a short, professional B2B sales quote email.
Keep the tone polished, realistic, and concise.

RFQ details:
- RFQ ID: {row.get("rfq_id")}
- Customer: {row.get("customer_name")}
- Part: {row.get("description")}
- Quantity: {row.get("quantity")}
- Unit quote price: {_money(row.get("unit_quote_price"))}
- Total quote value: {_money(row.get("total_quote_value"))}
- Estimated lead time: {row.get("lead_time_days", row.get("requested_lead_time_days", "TBD"))} business days
- Win probability: {_percent(row.get("win_probability"))}
- Recommended action: {row.get("recommended_action")}
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sales operations analyst writing professional quote emails for B2B customers.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def generate_ai_email(row: pd.Series | dict[str, Any]) -> str:
    """Use OpenAI when configured, otherwise return the template email."""
    ai_email = _openai_email(row)
    if ai_email:
        return ai_email
    return generate_template_email(row)


def generate_email_with_source(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    ai_email = _openai_email(row)
    if ai_email:
        return ai_email, "OpenAI API"
    return generate_template_email(row), "Template fallback"


def add_emails_to_quotes(df: pd.DataFrame, use_openai: bool = False) -> pd.DataFrame:
    """Add quote emails to final records.

    Batch exports default to the template path so GitHub and Streamlit demos run
    quickly without requiring paid API calls. The dashboard can still generate an
    OpenAI-assisted email for a selected RFQ through generate_email_with_source.
    """
    quotes = df.copy()
    if use_openai:
        generated = quotes.apply(lambda row: generate_email_with_source(row), axis=1)
    else:
        generated = quotes.apply(lambda row: (generate_template_email(row), "Template fallback"), axis=1)

    quotes["quote_email"] = generated.apply(lambda item: item[0])
    quotes["email_source"] = generated.apply(lambda item: item[1])
    return quotes
