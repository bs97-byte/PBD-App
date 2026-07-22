# Privacy by Design Assessment Studio

A rule-based Streamlit tool that scores any product/feature requirement against
Ann Cavoukian's 7 Foundational Principles of Privacy by Design, generates a
prioritized risk register, maps likely applicable regulations, and exports an
audit-ready report.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Quick start

1. In the sidebar, pick a sample requirement from the dropdown (or write your own).
2. Click **Scan text for data types** to auto-suggest data elements mentioned in the text.
3. Fill in the rest of the processing details (purpose, sharing, retention, consent, security, rights).
4. Click **Run Privacy Assessment**.
5. Explore the **Overview**, **Principle Deep-Dive**, **Risk Register**, **Regulatory
   Mapping**, **Export Report**, and **History** tabs.

## Extending the rules engine

Everything the engine "knows" lives in plain Python dictionaries and small
functions near the top of `app.py`:

- `DATA_ELEMENTS` — add a new data type and its sensitivity/category.
- `KEYWORD_MAP` — teach the text scanner to recognize a new data type.
- `p1_proactive` ... `p7_respect` — each is one Privacy by Design principle's
  rule set; add, remove, or reweight checks (severity drives the score delta
  via `SEV_DELTA`).
- `map_regulations` — add a new regulation and its trigger condition. Currently covers
  India's DPDP, EU/UK GDPR, California CCPA/CPRA, Brazil's LGPD, China's PIPL, HIPAA,
  children's privacy laws, and biometric privacy laws. FERPA (US education records) and
  state-level student-privacy laws (e.g., California SOPIPA) are natural next additions
  for education-sector use cases.

No retraining, no external API calls — every score is deterministic and
traceable to a specific rule, which matters for a tool whose job is to survive
an audit.
