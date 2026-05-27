from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from email_generator import generate_email_with_source
from email_generator import add_emails_to_quotes
from model_utils import ModelTrainingResult
from rfq_pipeline import (
    calculate_quote_price,
    load_data,
    merge_rfqs_with_catalog,
    prepare_model_features,
    save_final_quotes,
    score_current_rfqs,
)


PROJECT_DIR = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_DIR / "docs"

st.set_page_config(
    page_title="AI-Powered RFQ Automation Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1500px;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8e1ea;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }
    .portfolio-callout {
        border-left: 4px solid #2563eb;
        background: #f8fafc;
        padding: 0.9rem 1rem;
        border-radius: 6px;
        margin: 0.4rem 0 1rem 0;
    }
    .small-note {
        color: #475569;
        font-size: 0.92rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "TBD"
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "TBD"
    return f"{value * 100:.1f}%"


def read_doc(file_name: str) -> str:
    path = DOCS_DIR / file_name
    if not path.exists():
        return f"`docs/{file_name}` has not been created yet."
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def get_pipeline_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, ModelTrainingResult, dict, str]:
    rfqs, parts_catalog, historical_rfqs = load_data()
    merged = merge_rfqs_with_catalog(rfqs, parts_catalog)
    priced = calculate_quote_price(merged)
    model_result = prepare_model_features(historical_rfqs)
    scored = score_current_rfqs(priced, model_result)
    final_quotes = add_emails_to_quotes(scored, use_openai=False)
    output_path = save_final_quotes(final_quotes)

    metrics = {
        "accuracy": model_result.accuracy,
        "roc_auc": model_result.roc_auc,
        "training_sample_size": model_result.training_sample_size,
        "test_sample_size": model_result.test_sample_size,
    }
    return rfqs, parts_catalog, historical_rfqs, final_quotes, model_result, metrics, str(output_path)


def filtered_quotes(final_quotes: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Dashboard Filters")
    st.sidebar.caption("Filter the active analysis without changing the source CSV files.")

    urgency_options = sorted(final_quotes["urgency"].dropna().unique())
    segment_options = sorted(final_quotes["customer_segment"].dropna().unique())
    priority_options = sorted(final_quotes["priority_status"].dropna().unique())
    industry_options = sorted(final_quotes["industry"].dropna().unique())

    selected_urgency = st.sidebar.multiselect("Urgency", urgency_options, default=urgency_options)
    selected_segments = st.sidebar.multiselect("Customer segment", segment_options, default=segment_options)
    selected_priorities = st.sidebar.multiselect("Priority status", priority_options, default=priority_options)
    selected_industries = st.sidebar.multiselect("Industry", industry_options, default=industry_options)

    st.sidebar.divider()
    st.sidebar.markdown("**Data note**")
    st.sidebar.caption("All RFQ, customer, pricing, and historical outcome data is synthetic and educational.")

    return final_quotes[
        final_quotes["urgency"].isin(selected_urgency)
        & final_quotes["customer_segment"].isin(selected_segments)
        & final_quotes["priority_status"].isin(selected_priorities)
        & final_quotes["industry"].isin(selected_industries)
    ].copy()


def show_executive_summary(quotes: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Executive Summary")
    st.markdown(
        '<div class="portfolio-callout">A quote-to-cash analytics workflow for pricing, margin visibility, sales prioritization, and quote communication.</div>',
        unsafe_allow_html=True,
    )

    total_rfqs = len(quotes)
    total_quote_value = quotes["total_quote_value"].sum()
    average_win_probability = quotes["win_probability"].mean()
    high_priority_count = int((quotes["priority_status"] == "High Priority").sum())
    strategic_review_count = int((quotes["priority_status"] == "Strategic Review").sum())
    average_margin = quotes["estimated_gross_margin_pct"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total RFQs", f"{total_rfqs:,}")
    col2.metric("Total Quote Value", format_currency(total_quote_value))
    col3.metric("Avg Win Probability", format_percent(average_win_probability))
    col4.metric("Avg Estimated Margin", format_percent(average_margin))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("High Priority", f"{high_priority_count:,}")
    col6.metric("Strategic Review", f"{strategic_review_count:,}")
    col7.metric("Model Accuracy", format_percent(metrics["accuracy"]))
    col8.metric("ROC-AUC", "N/A" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}")

    st.caption(
        f"Model trained on {metrics['training_sample_size']:,} synthetic records and evaluated on {metrics['test_sample_size']:,} holdout records."
    )


def show_data_preview(rfqs: pd.DataFrame, parts_catalog: pd.DataFrame, historical_rfqs: pd.DataFrame) -> None:
    st.subheader("RFQ and Catalog Data Preview")
    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["RFQ Data Preview", "Parts Catalog Preview", "Historical RFQs"])
    with preview_tab1:
        st.dataframe(rfqs, use_container_width=True, hide_index=True)
    with preview_tab2:
        st.dataframe(parts_catalog, use_container_width=True, hide_index=True)
    with preview_tab3:
        st.dataframe(historical_rfqs.head(50), use_container_width=True, hide_index=True)


def show_quote_engine(quotes: pd.DataFrame) -> None:
    st.subheader("Automated Pricing Table")
    display_quotes = quotes.copy()
    display_quotes["estimated_gross_margin_pct"] = display_quotes["estimated_gross_margin_pct"] * 100
    display_quotes["win_probability"] = display_quotes["win_probability"] * 100

    visible_columns = [
        "rfq_id",
        "customer_name",
        "industry",
        "part_number",
        "description",
        "quantity",
        "urgency",
        "unit_quote_price",
        "total_quote_value",
        "estimated_gross_margin_pct",
        "win_probability",
        "priority_status",
        "recommended_action",
    ]

    st.dataframe(
        display_quotes[visible_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "unit_quote_price": st.column_config.NumberColumn("Unit Price", format="$%.2f"),
            "total_quote_value": st.column_config.NumberColumn("Quote Value", format="$%.2f"),
            "estimated_gross_margin_pct": st.column_config.ProgressColumn(
                "Est. Margin", format="%.1f%%", min_value=0, max_value=100
            ),
            "win_probability": st.column_config.ProgressColumn(
                "Win Probability", format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )

    with st.expander("Pricing Rule Notes"):
        st.dataframe(
            quotes[["rfq_id", "customer_name", "part_number", "pricing_notes"]],
            use_container_width=True,
            hide_index=True,
        )


def show_scoring_summary(quotes: pd.DataFrame) -> None:
    st.subheader("Win Probability Scoring")
    display_quotes = quotes.copy()
    display_quotes["win_probability"] = display_quotes["win_probability"] * 100

    score_cols = [
        "rfq_id",
        "customer_name",
        "customer_segment",
        "urgency",
        "total_quote_value",
        "win_probability",
        "priority_status",
        "recommended_action",
    ]
    scored = display_quotes[score_cols].sort_values(["priority_status", "win_probability"], ascending=[True, False])
    st.dataframe(
        scored,
        use_container_width=True,
        hide_index=True,
        column_config={
            "total_quote_value": st.column_config.NumberColumn("Quote Value", format="$%.2f"),
            "win_probability": st.column_config.ProgressColumn(
                "Win Probability", format="%.1f%%", min_value=0, max_value=100
            ),
        },
    )

    st.markdown(
        """
        **Priority status definitions**

        - High Priority: win probability is at least 70%.
        - Sales Review: win probability is between 40% and 69%.
        - Strategic Review: win probability is below 40%.
        """
    )


def show_analytics(quotes: pd.DataFrame) -> None:
    st.subheader("Quote Intelligence Charts")
    color_sequence = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626", "#0891b2"]

    quote_value = quotes.sort_values("total_quote_value", ascending=False)
    urgency_count = quotes.groupby("urgency", as_index=False)["rfq_id"].count()
    segment_value = quotes.groupby("customer_segment", as_index=False)["total_quote_value"].sum()
    priority_count = quotes.groupby("priority_status", as_index=False)["rfq_id"].count()

    chart_col1, chart_col2 = st.columns(2)
    chart_col1.plotly_chart(
        px.bar(
            quote_value,
            x="rfq_id",
            y="total_quote_value",
            color="priority_status",
            title="Quote Value by RFQ",
            labels={"rfq_id": "RFQ ID", "total_quote_value": "Quote Value", "priority_status": "Priority"},
            color_discrete_sequence=color_sequence,
            template="plotly_white",
        ),
        use_container_width=True,
    )
    chart_col2.plotly_chart(
        px.bar(
            urgency_count,
            x="urgency",
            y="rfq_id",
            color="urgency",
            title="RFQ Count by Urgency",
            labels={"rfq_id": "RFQ Count", "urgency": "Urgency"},
            color_discrete_sequence=color_sequence,
            template="plotly_white",
        ),
        use_container_width=True,
    )

    chart_col3, chart_col4 = st.columns(2)
    chart_col3.plotly_chart(
        px.histogram(
            quotes,
            x="win_probability",
            nbins=8,
            color="priority_status",
            title="Win Probability Distribution",
            labels={"win_probability": "Win Probability", "priority_status": "Priority"},
            color_discrete_sequence=color_sequence,
            template="plotly_white",
        ),
        use_container_width=True,
    )
    chart_col4.plotly_chart(
        px.pie(
            segment_value,
            values="total_quote_value",
            names="customer_segment",
            title="Quote Value by Customer Segment",
            color_discrete_sequence=color_sequence,
            template="plotly_white",
        ),
        use_container_width=True,
    )

    chart_col5, chart_col6 = st.columns(2)
    chart_col5.plotly_chart(
        px.bar(
            priority_count,
            x="priority_status",
            y="rfq_id",
            color="priority_status",
            title="Priority Status Mix",
            labels={"priority_status": "Priority Status", "rfq_id": "RFQ Count"},
            color_discrete_sequence=color_sequence,
            template="plotly_white",
        ),
        use_container_width=True,
    )

    risk_view = quotes[
        [
            "customer_name",
            "part_category",
            "inventory_level",
            "quantity",
            "supplier_risk_score",
            "total_quote_value",
        ]
    ].copy()
    risk_view["inventory_gap"] = risk_view["inventory_level"] - risk_view["quantity"]
    chart_col6.plotly_chart(
        px.scatter(
            risk_view,
            x="supplier_risk_score",
            y="inventory_gap",
            size="total_quote_value",
            color="part_category",
            hover_name="customer_name",
            title="Inventory and Supplier Risk View",
            labels={"supplier_risk_score": "Supplier Risk Score", "inventory_gap": "Inventory Surplus / Gap"},
            color_discrete_sequence=color_sequence,
            template="plotly_white",
        ),
        use_container_width=True,
    )


def show_email_generator(quotes: pd.DataFrame) -> None:
    st.subheader("Quote Email Generator")
    st.caption("Uses OpenAI when an API key is configured, otherwise generates a professional template email.")

    selected_rfq = st.selectbox("Select RFQ ID", quotes["rfq_id"].tolist())
    selected_row = quotes.loc[quotes["rfq_id"] == selected_rfq].iloc[0]
    email_body, email_source = generate_email_with_source(selected_row)

    detail_col, email_col = st.columns([1, 2])
    with detail_col:
        st.markdown("#### Key Quote Details")
        st.write(f"Customer: **{selected_row['customer_name']}**")
        st.write(f"Part: **{selected_row['description']}**")
        st.write(f"Quantity: **{selected_row['quantity']:,}**")
        st.write(f"Unit price: **{format_currency(selected_row['unit_quote_price'])}**")
        st.write(f"Total value: **{format_currency(selected_row['total_quote_value'])}**")
        st.write(f"Win probability: **{format_percent(selected_row['win_probability'])}**")
        st.info(f"Email source: {email_source}")

    with email_col:
        st.text_area("Email body", email_body, height=360)


def build_simulated_quote(
    rfq_input: dict[str, object],
    parts_catalog: pd.DataFrame,
    model_result: ModelTrainingResult,
) -> pd.DataFrame:
    """Run a single entered RFQ through the same dashboard pricing and scoring pipeline."""
    rfq_df = pd.DataFrame([rfq_input])
    merged = merge_rfqs_with_catalog(rfq_df, parts_catalog)
    priced = calculate_quote_price(merged)
    scored = score_current_rfqs(priced, model_result)
    email_body, email_source = generate_email_with_source(scored.iloc[0])
    scored["quote_email"] = email_body
    scored["email_source"] = email_source
    return scored


def show_new_rfq_simulator(
    rfqs: pd.DataFrame,
    parts_catalog: pd.DataFrame,
    model_result: ModelTrainingResult,
) -> None:
    st.subheader("New RFQ Simulator")
    st.caption(
        "Enter a new synthetic RFQ and generate a live quote recommendation without changing any source CSV files."
    )

    segment_options = sorted(rfqs["customer_segment"].dropna().unique())
    industry_options = sorted(rfqs["industry"].dropna().unique())
    region_options = sorted(rfqs["region"].dropna().unique())
    part_options = sorted(parts_catalog["part_number"].dropna().unique())
    payment_options = sorted(rfqs["payment_terms"].dropna().unique())
    tier_options = sorted(rfqs["customer_tier"].dropna().unique())

    with st.form("new_rfq_simulator_form", clear_on_submit=False):
        form_col1, form_col2, form_col3 = st.columns(3)

        with form_col1:
            customer_name = st.text_input("Customer Name", value="Example Manufacturing Co.")
            customer_segment = st.selectbox("Customer Segment", segment_options)
            industry = st.selectbox("Industry", industry_options)

        with form_col2:
            region = st.selectbox("Region", region_options)
            part_number = st.selectbox("Part Number", part_options)
            quantity = st.number_input("Quantity", min_value=1, max_value=10000, value=75, step=1)

        with form_col3:
            urgency = st.selectbox("Urgency", ["Low", "Medium", "High"], index=1)
            requested_lead_time_days = st.number_input(
                "Requested Lead Time Days", min_value=1, max_value=365, value=14, step=1
            )
            payment_terms = st.selectbox("Payment Terms", payment_options)
            customer_tier = st.selectbox("Customer Tier", tier_options)

        submitted = st.form_submit_button("Generate RFQ Result", use_container_width=True)

    if not submitted:
        st.info("Complete the form and click Generate RFQ Result to simulate pricing and sales prioritization.")
        return

    rfq_input = {
        "rfq_id": f"SIM-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}",
        "customer_name": customer_name.strip() or "New Customer",
        "customer_segment": customer_segment,
        "industry": industry,
        "region": region,
        "part_number": part_number,
        "quantity": int(quantity),
        "urgency": urgency,
        "requested_lead_time_days": int(requested_lead_time_days),
        "payment_terms": payment_terms,
        "customer_tier": customer_tier,
    }

    simulated_quote = build_simulated_quote(rfq_input, parts_catalog, model_result)
    result = simulated_quote.iloc[0]

    st.markdown("#### Simulated RFQ Result")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Unit Quote Price", format_currency(result["unit_quote_price"]))
    kpi2.metric("Total Quote Value", format_currency(result["total_quote_value"]))
    kpi3.metric("Win Probability", format_percent(result["win_probability"]))
    kpi4.metric("Estimated Margin", format_percent(result["estimated_gross_margin_pct"]))
    kpi5.metric("Priority Status", result["priority_status"])

    detail_col, email_col = st.columns([1, 2])
    with detail_col:
        st.markdown("#### Sales Guidance")
        st.write(f"**Recommended action:** {result['recommended_action']}")
        st.write(f"**Pricing notes:** {result['pricing_notes']}")
        st.write(f"**Email source:** {result['email_source']}")

        download_columns = [
            "rfq_id",
            "customer_name",
            "customer_segment",
            "industry",
            "region",
            "part_number",
            "description",
            "quantity",
            "urgency",
            "requested_lead_time_days",
            "payment_terms",
            "customer_tier",
            "unit_quote_price",
            "total_quote_value",
            "estimated_gross_margin_pct",
            "win_probability",
            "priority_status",
            "recommended_action",
            "pricing_notes",
            "quote_email",
            "email_source",
        ]
        download_df = simulated_quote[download_columns]
        st.download_button(
            label="Download simulated RFQ CSV",
            data=download_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{result['rfq_id']}_simulated_quote.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with email_col:
        st.markdown("#### Generated Quote Email")
        st.text_area("Simulated quote email", result["quote_email"], height=360)


def show_export_center(quotes: pd.DataFrame, output_path: str) -> None:
    st.subheader("Final CSV Download")
    csv_bytes = quotes.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download final_quotes.csv",
        data=csv_bytes,
        file_name="final_quotes.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.success(f"Final quotes are also saved locally at: {output_path}")


def show_resume_summary() -> None:
    st.subheader("Resume and Interview Summary")
    resume_tab, interview_tab = st.tabs(["Resume Bullets", "Interview Talking Points"])
    with resume_tab:
        st.markdown(read_doc("resume_bullets.md"))
    with interview_tab:
        st.markdown(read_doc("interview_talking_points.md"))


def main() -> None:
    st.title("AI-Powered RFQ Automation & Sales Quote Intelligence Dashboard")
    st.caption(
        "A professional portfolio project for RFQ automation, pricing analytics, sales prioritization, and AI-assisted quote communication."
    )

    rfqs, parts_catalog, historical_rfqs, final_quotes, model_result, metrics, output_path = get_pipeline_outputs()
    quotes = filtered_quotes(final_quotes)

    if quotes.empty:
        st.warning("No RFQs match the current filters. Adjust the sidebar filters to restore dashboard results.")
        show_data_preview(rfqs, parts_catalog, historical_rfqs)
        return

    tabs = st.tabs(
        [
            "Executive Summary",
            "Data Preview",
            "Pricing Table",
            "Win Scoring",
            "New RFQ Simulator",
            "Charts",
            "Email Generator",
            "CSV Download",
            "Resume Summary",
        ]
    )

    with tabs[0]:
        show_executive_summary(quotes, metrics)
    with tabs[1]:
        show_data_preview(rfqs, parts_catalog, historical_rfqs)
    with tabs[2]:
        show_quote_engine(quotes)
    with tabs[3]:
        show_scoring_summary(quotes)
    with tabs[4]:
        show_new_rfq_simulator(rfqs, parts_catalog, model_result)
    with tabs[5]:
        show_analytics(quotes)
    with tabs[6]:
        show_email_generator(final_quotes)
    with tabs[7]:
        show_export_center(final_quotes, output_path)
    with tabs[8]:
        show_resume_summary()


if __name__ == "__main__":
    main()
