# Interview Talking Points

## 30-Second Explanation

I built an AI-powered RFQ automation dashboard that simulates a quote-to-cash sales operations workflow. It imports synthetic customer RFQs, matches requested parts to a catalog, applies pricing rules, predicts win probability, prioritizes opportunities, lets users enter a new RFQ live, and generates quote emails using either OpenAI or a professional fallback template. The goal is to show how analytics and AI can improve pricing consistency, margin visibility, and sales response time.

## 1-Minute Explanation

This project is a business analytics and sales operations automation system built with Python, Pandas, scikit-learn, Streamlit, Plotly, and optional OpenAI API support. I designed synthetic RFQ, catalog, and historical sales data to avoid confidential information while still modeling a realistic B2B workflow. The pipeline joins customer requests to part data, flags unmatched items, calculates quote prices using urgency, volume, customer tier, supplier risk, and inventory rules, then trains a machine learning model to estimate win probability. The Streamlit dashboard gives business users KPI cards, quote tables, sales prioritization, margin views, risk charts, export functionality, and generated quote emails.

## Technical Explanation

The project uses a modular Python structure:

- `data_generator.py` creates reproducible synthetic CSV data with a fixed random seed.
- `rfq_pipeline.py` loads and validates data, merges RFQs with the parts catalog, applies pricing rules, scores current opportunities, and saves `final_quotes.csv`.
- `model_utils.py` trains a scikit-learn pipeline with `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, and `LogisticRegression`.
- `email_generator.py` uses OpenAI only when `OPENAI_API_KEY` exists and automatically falls back to a professional template.
- `app.py` provides the Streamlit dashboard, filters, charts, model metrics, New RFQ Simulator, download buttons, email generator, and resume summary.

## Business Value Explanation

The project demonstrates how analytics can improve an RFQ workflow by reducing manual effort, improving pricing consistency, highlighting margin impact, prioritizing high-probability opportunities, and making customer communication faster and more standardized.

It is relevant to business analyst, data analyst, FP&A, sales operations, ERP systems, and AI automation roles because it connects operational data, pricing rules, predictive modeling, dashboard storytelling, and workflow automation.

## Possible Interview Questions

### What business problem does this solve?

Companies often handle RFQs manually, which can slow response time and create inconsistent pricing. This dashboard automates the workflow and gives sales, finance, and operations users better visibility into quote value, margin, risk, and win probability.

### Why did you use synthetic data?

RFQ, pricing, customer, supplier, and margin data can be confidential. I used synthetic data so the project is safe to publish on GitHub while still representing realistic B2B sales operations patterns.

### How does the pricing engine work?

The pricing engine starts with catalog base price and applies rules for urgency, volume discounts, strategic customer relationships, supplier risk, and inventory constraints. It then calculates unit quote price, total quote value, estimated gross margin, and pricing notes.

### How does the win probability model work?

The model is trained on synthetic historical RFQ records. It uses categorical features such as customer segment, industry, part category, urgency, and customer tier, plus numeric features such as order size, lead time, price competitiveness, supplier risk, and gross margin. The output is a probability score used to prioritize opportunities.

### How does the app work without OpenAI?

OpenAI is optional. If no `OPENAI_API_KEY` is configured, the app uses a deterministic template email generator. This makes the project safe for GitHub, Streamlit Community Cloud, recruiter demos, and classroom review.

### What would you improve in a real company deployment?

I would connect the workflow to ERP and CRM systems, add approval controls for discounting, use real historical outcomes after data governance review, monitor model performance, add authentication, and create audit logs for pricing decisions.

### How does this relate to FP&A or finance analytics?

It supports margin visibility, price governance, forecast prioritization, sales pipeline quality, and decision support. Quoting decisions influence revenue, gross margin, working capital, and demand planning, which are all relevant to finance analytics.
