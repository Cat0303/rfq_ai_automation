# AI-Powered RFQ Automation & Sales Quote Intelligence Dashboard

A professional Streamlit dashboard project that turns synthetic customer RFQs into priced, prioritized, and email-ready sales quote recommendations.

This project is designed for a resume, portfolio, GitHub, and Streamlit Community Cloud demo. It shows how Python, analytics, machine learning, and optional generative AI can support a realistic quote-to-cash business workflow.

## Project Highlights

- Executive KPI cards for RFQ count, quote value, margin, win probability, and model performance
- RFQ and parts catalog previews
- Automated pricing table with dynamic pricing rules
- Win probability scoring using scikit-learn
- Sales priority labels: High Priority, Sales Review, and Strategic Review
- Plotly charts for quote value, urgency, win probability, customer segment, priority mix, and supplier risk
- Quote email generator with optional OpenAI support and a template fallback
- Download button for `final_quotes.csv`
- Resume bullets and interview talking points included in `docs/`

## Business Problem

Sales and operations teams often process RFQs manually. That can slow response time, create inconsistent pricing, reduce margin visibility, and make it harder to identify the most valuable opportunities.

This dashboard simulates a practical RFQ automation workflow where incoming customer requests are matched to a parts catalog, priced with business rules, scored for win probability, prioritized for sales follow-up, and converted into professional quote emails.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- python-dotenv
- OpenAI API, optional only

## Repository Structure

```text
rfq-ai-automation/
|
|-- app.py
|-- rfq_pipeline.py
|-- email_generator.py
|-- data_generator.py
|-- model_utils.py
|-- requirements.txt
|-- README.md
|-- .env.example
|-- .gitignore
|
|-- data/
|   |-- rfqs.csv
|   |-- parts_catalog.csv
|   |-- historical_rfqs.csv
|
|-- outputs/
|   |-- .gitkeep
|   |-- final_quotes.csv
|
|-- docs/
|   |-- resume_bullets.md
|   |-- interview_talking_points.md
|
|-- screenshots/
|   |-- README_PLACEHOLDER.txt
```

## Data

All data is synthetic and educational. The project does not contain real customer, pricing, supplier, or transaction information.

Input files:

- `data/rfqs.csv` - current customer RFQ requests
- `data/parts_catalog.csv` - part descriptions, cost, pricing, inventory, lead time, and supplier risk
- `data/historical_rfqs.csv` - synthetic historical quote outcomes used for model training

Output file:

- `outputs/final_quotes.csv` - final priced, scored, prioritized, and email-ready quote output

The app can also read root-level CSV files with the same names for compatibility with earlier project versions.

## How the Pipeline Works

1. Load and validate RFQ, catalog, and historical RFQ CSV files.
2. Match current RFQs to the parts catalog by `part_number`.
3. Flag unmatched parts for manual review.
4. Apply pricing rules for urgency, volume, customer tier, supplier risk, and inventory constraints.
5. Calculate unit quote price, total quote value, and estimated gross margin.
6. Train a scikit-learn logistic regression model on historical RFQ outcomes.
7. Predict win probability for current RFQs.
8. Assign priority status and recommended sales actions.
9. Generate quote emails with a template fallback, or OpenAI when configured.
10. Save and serve `final_quotes.csv` for download.

## Pricing Logic

The pricing engine starts with catalog base price and applies:

- High urgency surcharge: +10%
- Medium urgency surcharge: +5%
- Quantity over 100 discount: -5%
- Quantity over 250 discount: -8%
- Strategic customer discount: -3%
- Supplier risk surcharge when risk is above 0.70: +4%
- Inventory constraint surcharge when requested quantity exceeds inventory: +6%

## Win Probability Model

The project uses a scikit-learn pipeline with:

- `ColumnTransformer`
- `OneHotEncoder`
- `StandardScaler`
- `LogisticRegression`
- Train/test split
- Accuracy and ROC-AUC reporting

Model features include customer segment, industry, part category, urgency, customer tier, order size, lead time, price competitiveness, supplier risk score, and gross margin percentage.

Priority rules:

- `High Priority` - win probability is at least 70%
- `Sales Review` - win probability is 40% to 69%
- `Strategic Review` - win probability is below 40%

## Quote Email Generator

The dashboard works without OpenAI. If no API key is available, it uses a professional template-based email generator.

OpenAI is optional. When `OPENAI_API_KEY` is configured locally or in Streamlit secrets, the email panel can generate an AI-assisted email for the selected RFQ.

No API keys are hardcoded. Do not commit `.env`, `config.txt`, or Streamlit secrets files.

## Local Run Instructions

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional virtual environment setup:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source .venv/bin/activate
```

## Optional OpenAI Setup

Create a local `.env` file:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

For Streamlit Community Cloud, add this value under app secrets instead of committing it to GitHub:

```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

The app still runs fully without this key.

## Streamlit Community Cloud Deployment

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Select the GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy the app.
6. Add `OPENAI_API_KEY` as an optional secret only if you want AI-generated emails.

## Resume Positioning

Suggested project title:

**AI-Powered RFQ Automation & Sales Quote Intelligence Dashboard**

Suggested resume line:

Built a Streamlit RFQ automation dashboard using Python, Pandas, scikit-learn, and Plotly to match customer quote requests with parts catalog data, apply pricing rules, predict win probability, prioritize sales opportunities, and generate quote-ready customer emails.

## Interview Summary

This project demonstrates how analytics can improve a quote-to-cash workflow by reducing manual effort, standardizing pricing, surfacing margin visibility, helping sales teams prioritize opportunities, and accelerating customer communication.

It is especially relevant for business analyst, data analyst, FP&A, sales operations, ERP systems, and AI automation roles because it connects technical implementation to measurable business process value.

## Future Improvements

- Connect to CRM or ERP data after governance review
- Add approval workflows for low-margin or high-discount quotes
- Add sales rep assignment and SLA tracking
- Add model monitoring and drift checks
- Add authentication and role-based views
- Add audit logs for pricing decisions
- Add scenario analysis for discounting and lead time changes
