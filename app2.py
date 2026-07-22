"""
Privacy by Design Assessment Studio
------------------------------------
A rule-based, transparent engine that scores any product/feature requirement
against Ann Cavoukian's 7 Foundational Principles of Privacy by Design,
surfaces a prioritized risk register, maps likely applicable regulations,
and produces an exportable, audit-ready report.

Design intent: deterministic and explainable (no black-box ML scoring),
runs entirely locally (no requirement text leaves the machine), and is
built to be extended (new principles, regulations, or data types are
single dictionary edits away).
"""

import json
import re
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# 1. DOMAIN KNOWLEDGE (the "rules engine" data — edit these to extend the tool)
# ============================================================================

DATA_ELEMENTS = {
    "Full Name": {"sensitivity": 2, "category": "Identifier"},
    "Email Address": {"sensitivity": 2, "category": "Identifier"},
    "Phone Number": {"sensitivity": 2, "category": "Identifier"},
    "Physical Address": {"sensitivity": 2, "category": "Identifier"},
    "Government ID / SSN / Aadhaar": {"sensitivity": 5, "category": "Special Category"},
    "Financial / Payment Data": {"sensitivity": 4, "category": "Financial"},
    "Health / Medical Records": {"sensitivity": 5, "category": "Health"},
    "Biometric Data": {"sensitivity": 5, "category": "Biometric/Genetic"},
    "Genetic Data": {"sensitivity": 5, "category": "Biometric/Genetic"},
    "Precise Geolocation": {"sensitivity": 3, "category": "Behavioral"},
    "Behavioral / Tracking Data": {"sensitivity": 3, "category": "Behavioral"},
    "Device / IP Address": {"sensitivity": 2, "category": "Technical"},
    "Children's Data (<18)": {"sensitivity": 5, "category": "Vulnerable Group"},
    "Race / Ethnicity / Religion / Political Opinion": {"sensitivity": 5, "category": "Special Category"},
    "Sexual Orientation": {"sensitivity": 5, "category": "Special Category"},
    "Employment / HR Data": {"sensitivity": 2, "category": "Identifier"},
    "Voice / Audio Recordings": {"sensitivity": 3, "category": "Biometric/Genetic"},
}

KEYWORD_MAP = {
    "Email Address": [r"\bemail\b"],
    "Phone Number": [r"\bphone number\b", r"\bmobile number\b", r"\bcontact number\b"],
    "Physical Address": [r"\bhome address\b", r"\bpostal address\b", r"\bmailing address\b"],
    "Government ID / SSN / Aadhaar": [r"\bssn\b", r"\bsocial security\b", r"\baadhaar\b", r"\bpassport number\b", r"\bnational id\b"],
    "Financial / Payment Data": [r"\bcredit card\b", r"\bdebit card\b", r"\bbank account\b", r"\bupi\b", r"\bpayment\b", r"\btransaction\b"],
    "Health / Medical Records": [r"\bhealth\b", r"\bmedical\b", r"\bdiagnosis\b", r"\bpatient\b", r"\bprescription\b", r"\btreatment\b"],
    "Biometric Data": [r"\bbiometric\b", r"\bfingerprint\b", r"\bface id\b", r"\bfacial recognition\b", r"\bretina\b", r"\biris scan\b"],
    "Genetic Data": [r"\bgenetic\b", r"\bdna\b"],
    "Precise Geolocation": [r"\blocation\b", r"\bgps\b", r"\bgeolocation\b"],
    "Behavioral / Tracking Data": [r"\bbrowsing\b", r"\bclickstream\b", r"\btracking\b", r"\bcookies\b", r"\bbehavior"],
    "Device / IP Address": [r"\bip address\b", r"\bdevice id\b", r"\bmac address\b"],
    "Children's Data (<18)": [r"\bchild", r"\bminor", r"\bkid", r"\bunder 13\b", r"\bunder 18\b", r"\bstudent"],
    "Race / Ethnicity / Religion / Political Opinion": [r"\brace\b", r"\bethnicity\b", r"\breligion\b", r"\bcaste\b", r"\bpolitical\b"],
    "Sexual Orientation": [r"\bsexual orientation\b", r"\blgbtq\b"],
    "Employment / HR Data": [r"\bemployee\b", r"\bpayroll\b", r"\bhr data\b", r"\bemployment\b"],
    "Voice / Audio Recordings": [r"\bvoice\b", r"\baudio recording\b", r"\bcall recording\b"],
    "Full Name": [r"\bfull name\b", r"\bfirst name\b", r"\blast name\b"],
}

GEOGRAPHY_OPTIONS = ["India", "EU/EEA", "United Kingdom", "United States - California", "United States - Other", "Brazil", "China", "Global / Other"]
SUBJECT_OPTIONS = ["Customers / End Users", "Employees", "Minors / Children", "Patients", "Vulnerable Groups (elderly, disabled, etc.)", "General Public"]
SHARING_OPTIONS = ["None", "Domestic vendors only", "International transfer", "Public / Open dataset"]
RETENTION_OPTIONS = ["Yes - defined period", "No - indefinite / not yet decided"]
CONSENT_OPTIONS = ["Explicit opt-in", "Opt-out", "Legitimate interest / necessity", "Not yet defined"]
SECURITY_OPTIONS = ["Encryption at rest", "Encryption in transit", "Access controls / RBAC", "Anonymization", "Pseudonymization", "Audit logging", "None yet"]
DSR_OPTIONS = ["Access", "Deletion / Erasure", "Portability", "Rectification", "Objection / Opt-out", "None yet"]

SEV_DELTA = {"High": -30, "Medium": -15, "Low": -8}
SEV_RANK = {"High": 0, "Medium": 1, "Low": 2}

SAMPLE_REQUIREMENTS = {
    "-- Select a sample --": "",
    "Higher-risk: location-based marketing app": (
        "Build a mobile app that captures the user's precise GPS location every 5 minutes "
        "in the background to send personalized coupons. We also store the user's phone "
        "number and email for marketing campaigns. There is currently no age gate. Data is "
        "kept indefinitely to improve our recommendation models."
    ),
    "Lower-risk: internal HR directory": (
        "An internal employee directory showing full name, work email, department and "
        "employment start date, visible only to logged-in employees on the corporate intranet."
    ),
    "Health context: telemedicine appointment booking": (
        "A telemedicine feature where patients enter their name, phone number, and a brief "
        "description of their medical symptoms to book a video consultation with a doctor. "
        "Consultation recordings are stored for quality review."
    ),
}


# ============================================================================
# 2. PII / DATA-TYPE HINT DETECTION (lightweight NLP-free heuristic scan)
# ============================================================================

def detect_pii_hints(text: str):
    if not text:
        return []
    t = text.lower()
    hits = []
    for element, patterns in KEYWORD_MAP.items():
        for p in patterns:
            if re.search(p, t):
                hits.append(element)
                break
    return hits


def mk_issue(finding: str, severity: str, recommendation: str):
    return {"finding": finding, "severity": severity, "recommendation": recommendation}


# ============================================================================
# 3. PRIVACY BY DESIGN PRINCIPLE RULE ENGINE
#    Each function inspects the structured requirement and returns
#    (score 0-100, list_of_issues). Deterministic and fully explainable.
# ============================================================================

def p1_proactive(inp):
    score, issues = 100, []
    special = any(DATA_ELEMENTS[e]["sensitivity"] >= 4 for e in inp["data_elements"])
    if special and inp["dpia_conducted"] != "Yes":
        issues.append(mk_issue(
            "High-sensitivity / special-category data is involved, but no DPIA/PIA has been conducted.",
            "High", "Conduct a formal Data Protection / Privacy Impact Assessment before development begins."))
    if inp["dpia_conducted"] == "Not sure":
        issues.append(mk_issue(
            "DPIA status for this requirement is undocumented.",
            "Medium", "Confirm with the privacy/legal team whether a DPIA is mandated and record the outcome."))
    if (not inp["security_measures"]) or ("None yet" in inp["security_measures"]):
        issues.append(mk_issue(
            "No technical safeguards have been defined at the requirement stage.",
            "High", "Define encryption, access-control and minimization measures during design, not after an incident."))
    for iss in issues:
        score += SEV_DELTA[iss["severity"]]
    return max(0, score), issues


def p2_default(inp):
    score, issues = 100, []
    n = len(inp["data_elements"])
    if n >= 8:
        issues.append(mk_issue(
            f"{n} distinct data elements are being collected — a broad footprint for one feature.",
            "Medium", "Re-validate each field's necessity against the stated purpose (data minimization) and drop non-essential fields."))
    if inp["consent_mechanism"] == "Opt-out":
        issues.append(mk_issue(
            "Consent is opt-out rather than opt-in, so privacy-protective behavior is not the default.",
            "Medium", "Switch to opt-in consent, particularly for non-essential processing such as marketing or analytics."))
    elif inp["consent_mechanism"] == "Not yet defined":
        issues.append(mk_issue(
            "No consent mechanism has been defined for this processing.",
            "High", "Define a lawful basis and, where required, an explicit opt-in consent flow before collection begins."))
    has_children = "Children's Data (<18)" in inp["data_elements"] or "Minors / Children" in inp["data_subjects"]
    if has_children and inp["consent_mechanism"] != "Explicit opt-in":
        issues.append(mk_issue(
            "Children's data is involved without an explicit (parental/guardian) consent mechanism.",
            "High", "Implement verifiable parental/guardian consent before collecting or processing children's data."))
    for iss in issues:
        score += SEV_DELTA[iss["severity"]]
    return max(0, score), issues


def p3_embedded(inp):
    score, issues = 100, []
    sm = inp["security_measures"]
    if not any(x in sm for x in ["Pseudonymization", "Anonymization"]):
        issues.append(mk_issue(
            "No pseudonymization or anonymization technique is planned.",
            "Medium", "Evaluate pseudonymizing or anonymizing fields that don't need to stay directly identifiable."))
    if "Access controls / RBAC" not in sm:
        issues.append(mk_issue(
            "Role-based access control is not yet planned for this data.",
            "Medium", "Design least-privilege, role-based access control into the data architecture from the start."))
    if inp["automated_decision_making"] == "Yes":
        issues.append(mk_issue(
            "The requirement involves automated decision-making about individuals.",
            "Medium", "Embed a human-review path for automated decisions that materially affect people (e.g., GDPR Art. 22)."))
    for iss in issues:
        score += SEV_DELTA[iss["severity"]]
    return max(0, score), issues


def p4_full_functionality(inp):
    score, issues = 100, []
    if not inp["purpose"] or len(inp["purpose"].strip()) < 10:
        issues.append(mk_issue(
            "Business purpose is not clearly articulated.",
            "Medium", "Document a specific, well-defined business purpose so privacy and functionality can be balanced deliberately, not traded off."))
    if (not inp["security_measures"]) or ("None yet" in inp["security_measures"]):
        issues.append(mk_issue(
            "No privacy safeguard is currently paired with the stated business functionality.",
            "Medium", "Identify safeguards that let the feature work as intended while minimizing privacy impact (e.g., on-device processing, aggregation)."))
    for iss in issues:
        score += SEV_DELTA[iss["severity"]]
    return max(0, score), issues


def p5_security(inp):
    score, issues = 100, []
    sm = inp["security_measures"]
    if "Encryption at rest" not in sm:
        issues.append(mk_issue("Encryption at rest is not planned.", "High",
                                "Encrypt stored data, especially any special-category or financial data elements."))
    if "Encryption in transit" not in sm:
        issues.append(mk_issue("Encryption in transit is not planned.", "High",
                                "Enforce TLS/HTTPS and encrypted channels for all data transmission."))
    if "Audit logging" not in sm:
        issues.append(mk_issue("Audit logging is not planned.", "Low",
                                "Log access to sensitive data to support incident detection and forensic review."))
    if inp["retention_defined"] != "Yes - defined period":
        issues.append(mk_issue("No defined retention period — data may be kept indefinitely.", "High",
                                "Define a retention schedule and an automated deletion/archival process covering the full data lifecycle."))
    if inp["third_party_sharing"] == "International transfer":
        issues.append(mk_issue("Data is transferred internationally with no safeguard mentioned.", "Medium",
                                "Put a cross-border transfer mechanism in place (e.g., Standard Contractual Clauses, adequacy decision) before transfer."))
    for iss in issues:
        score += SEV_DELTA[iss["severity"]]
    return max(0, score), issues


def p6_transparency(inp):
    score, issues = 100, []
    if inp["third_party_sharing"] != "None":
        issues.append(mk_issue(
            f"Data is shared with third parties ({inp['third_party_sharing']}); this must be disclosed to users.",
            "Medium", "Disclose all third-party recipients and the purpose of sharing in the privacy notice."))
    if inp["automated_decision_making"] == "Yes":
        issues.append(mk_issue(
            "Automated decision-making is used but explainability to affected users isn't confirmed.",
            "Medium", "Provide users a plain-language explanation of the logic and consequences of the automated decision."))
    issues.append(mk_issue(
        "Reminder: confirm the public-facing privacy notice will be updated for this feature.",
        "Low", "Update the privacy notice/policy to describe this processing in plain language before launch."))
    for iss in issues:
        score += SEV_DELTA[iss["severity"]]
    return max(0, score), issues


def p7_respect(inp):
    score, issues = 100, []
    dsr = inp["dsr_support"]
    if (not dsr) or ("None yet" in dsr):
        issues.append(mk_issue(
            "No data-subject-rights fulfillment mechanism (access, deletion, etc.) is planned.",
            "High", "Build workflows for users to access, correct, delete, or export their data, and publish how to exercise these rights."))
    elif len(dsr) < 3:
        issues.append(mk_issue(
            f"Only {len(dsr)} data-subject right(s) are currently supported ({', '.join(dsr)}).",
            "Medium", "Extend coverage toward the full set of access, deletion, portability, rectification and objection rights."))
    special = any(DATA_ELEMENTS[e]["sensitivity"] >= 4 for e in inp["data_elements"])
    if special and inp["consent_mechanism"] != "Explicit opt-in":
        issues.append(mk_issue(
            "Special-category / high-sensitivity data is collected without explicit opt-in consent.",
            "High", "Require explicit, granular opt-in consent specifically for special-category data."))
    vulnerable = any(s in inp["data_subjects"] for s in ["Minors / Children", "Patients", "Vulnerable Groups (elderly, disabled, etc.)"])
    if vulnerable:
        issues.append(mk_issue(
            "Vulnerable data subjects (minors, patients, or similar) are involved.",
            "Low", "Apply heightened scrutiny and extra safeguards specifically for vulnerable data subjects."))
    for iss in issues:
        score += SEV_DELTA[iss["severity"]]
    return max(0, score), issues


PRINCIPLES = [
    ("proactive", "Proactive, Not Reactive", "Anticipates and prevents privacy risks before they occur, rather than remediating after the fact.", p1_proactive),
    ("default", "Privacy as the Default", "Maximum privacy protection is achieved automatically, without requiring user action.", p2_default),
    ("embedded", "Privacy Embedded in Design", "Privacy is a core architectural component of the system, not a bolt-on control.", p3_embedded),
    ("full_functionality", "Full Functionality (Positive-Sum)", "Accommodates business goals and privacy together rather than trading one off against the other.", p4_full_functionality),
    ("security", "End-to-End Security", "Protects data securely across its entire lifecycle, from collection to deletion.", p5_security),
    ("transparency", "Visibility and Transparency", "Practices are open, documented, and verifiable to users and stakeholders.", p6_transparency),
    ("respect", "Respect for User Privacy", "Keeps the interests of the individual data subject paramount.", p7_respect),
]


# ============================================================================
# 4. SCORING ORCHESTRATION
# ============================================================================

def run_assessment(inputs: dict, weights: dict):
    principle_scores = {}
    all_issues = []
    for key, short, desc, func in PRINCIPLES:
        score, issues = func(inputs)
        principle_scores[key] = {"name": short, "description": desc, "score": score}
        for iss in issues:
            enriched = dict(iss)
            enriched["principle"] = short
            all_issues.append(enriched)
    total_w = sum(weights.values()) or 1
    overall = sum(principle_scores[k]["score"] * weights[k] for k in principle_scores) / total_w
    return {"principle_scores": principle_scores, "issues": all_issues, "overall_score": round(overall, 1)}


def status_label(score):
    if score >= 80:
        return "Strong"
    if score >= 50:
        return "Needs Improvement"
    return "High Risk"


def status_emoji(score):
    if score >= 80:
        return "\U0001F7E2"
    if score >= 50:
        return "\U0001F7E1"
    return "\U0001F534"


# ============================================================================
# 5. REGULATORY MAPPING (heuristic, non-exhaustive, decision-support only)
# ============================================================================

def map_regulations(inputs: dict):
    regs = []
    geo, elems, subjects = inputs["geography"], inputs["data_elements"], inputs["data_subjects"]

    if "India" in geo:
        regs.append({
            "name": "India — Digital Personal Data Protection Act, 2023 (DPDP)",
            "reason": "Processing involves data principals located in India.",
            "obligations": [
                "Provide clear notice and obtain consent (or rely on a specified 'legitimate use') before processing.",
                "Obtain verifiable parental/guardian consent for children's or vulnerable persons' data.",
                "Honor rights to access, correction, erasure, and grievance redressal.",
                "Report significant personal data breaches to the Data Protection Board and affected users.",
            ],
        })
    if any(g in geo for g in ["EU/EEA", "United Kingdom"]):
        regs.append({
            "name": "EU GDPR / UK GDPR",
            "reason": "Processing involves data subjects in the EU/EEA or UK.",
            "obligations": [
                "Establish and document a lawful basis under Art. 6 (and an Art. 9 condition for special-category data).",
                "Conduct a DPIA under Art. 35 for likely high-risk processing.",
                "Support access, erasure, portability, rectification and objection rights within statutory timelines.",
                "Use an approved transfer mechanism (SCCs / adequacy) for any transfer outside the EEA/UK.",
                "Report qualifying breaches to the supervisory authority within 72 hours.",
            ],
        })
    if "United States - California" in geo:
        regs.append({
            "name": "California CCPA / CPRA",
            "reason": "Processing involves California residents.",
            "obligations": [
                "Provide notice at collection describing categories of data and purposes.",
                "Honor rights to know, delete, correct, and opt out of sale/sharing of personal information.",
                "Apply extra limits on use of Sensitive Personal Information if collected.",
                "Do not discriminate against users who exercise their privacy rights.",
            ],
        })
    if "Brazil" in geo:
        regs.append({
            "name": "Brazil — Lei Geral de Proteção de Dados (LGPD)",
            "reason": "Processing involves data subjects located in Brazil.",
            "obligations": [
                "Establish a legal basis for processing under Art. 7 (or Art. 11 for sensitive data).",
                "Appoint a Data Protection Officer (encarregado) as a point of contact for data subjects and ANPD.",
                "Conduct a Data Protection Impact Report (RIPD) for high-risk or sensitive-data processing.",
                "Obtain specific, prominent parental/guardian consent for children's and adolescents' data, applied in their best interest.",
                "Support rights to confirmation, access, correction, anonymization, deletion, and portability; report qualifying breaches to the ANPD.",
            ],
        })
    if "China" in geo:
        regs.append({
            "name": "China — Personal Information Protection Law (PIPL)",
            "reason": "Processing involves data subjects located in China.",
            "obligations": [
                "Obtain separate, explicit consent for sensitive personal information, cross-border transfer, public disclosure, and automated decision-making.",
                "Conduct a Personal Information Protection Impact Assessment (PIPIA) before these high-risk activities.",
                "Complete a CAC security assessment, certification, or standard contract before transferring data outside China (thresholds apply).",
                "Appoint a local representative/entity in China if processing China-based data subjects from abroad.",
                "Obtain guardian consent and apply heightened protection for data of minors under 14 (treated as sensitive personal information).",
            ],
        })
    if "Health / Medical Records" in elems:
        regs.append({
            "name": "HIPAA (if operating as a covered entity/business associate in the US)",
            "reason": "Health/medical data elements are collected.",
            "obligations": [
                "Apply the 'minimum necessary' standard to uses and disclosures of health data.",
                "Implement administrative, physical, and technical safeguards.",
                "Execute Business Associate Agreements with relevant vendors.",
                "Follow HIPAA breach notification timelines and procedures.",
            ],
        })
    if "Children's Data (<18)" in elems or "Minors / Children" in subjects:
        regs.append({
            "name": "Children's privacy laws (e.g., COPPA, DPDP Sec. 9, GDPR Art. 8)",
            "reason": "Children or minors are data subjects.",
            "obligations": [
                "Obtain verifiable parental/guardian consent before collecting children's data.",
                "Minimize data collection from children to what's strictly necessary.",
                "Avoid behavioral advertising or profiling of children without explicit consent.",
            ],
        })
    if "Biometric Data" in elems:
        regs.append({
            "name": "Biometric privacy laws (e.g., Illinois BIPA, GDPR Art. 9)",
            "reason": "Biometric data is collected.",
            "obligations": [
                "Obtain written/explicit consent before capturing biometric identifiers.",
                "Publish a retention schedule and a firm destruction timeline for biometric data.",
                "Never sell, lease, or profit from biometric data.",
            ],
        })
    return regs


# ============================================================================
# 6. REPORT GENERATION (Markdown + JSON, for audit trail / GRC integration)
# ============================================================================

def generate_markdown_report(project_name, timestamp, inputs, result, regulations):
    L = []
    L.append("# Privacy by Design Assessment Report")
    L.append(f"**Project / Feature:** {project_name or 'Untitled'}  ")
    L.append(f"**Assessed on:** {timestamp}  ")
    L.append(f"**Overall Privacy Score:** {result['overall_score']}/100 ({status_label(result['overall_score'])})\n")

    L.append("## Requirement Summary")
    L.append(inputs.get("requirement_text") or "_Not provided_")

    L.append("\n## Processing Details")
    L.append(f"- **Purpose:** {inputs['purpose'] or '_Not specified_'}")
    L.append(f"- **Data Elements:** {', '.join(inputs['data_elements']) or 'None specified'}")
    L.append(f"- **Data Subjects:** {', '.join(inputs['data_subjects']) or 'None specified'}")
    L.append(f"- **Geography:** {', '.join(inputs['geography']) or 'None specified'}")
    L.append(f"- **Third-Party Sharing:** {inputs['third_party_sharing']}")
    ret = inputs["retention_defined"] + (f" — {inputs['retention_period']}" if inputs.get("retention_period") else "")
    L.append(f"- **Retention:** {ret}")
    L.append(f"- **Consent Mechanism:** {inputs['consent_mechanism']}")
    L.append(f"- **Security Measures:** {', '.join(inputs['security_measures']) or 'None specified'}")
    L.append(f"- **Data Subject Rights Supported:** {', '.join(inputs['dsr_support']) or 'None specified'}")
    L.append(f"- **Automated Decision-Making:** {inputs['automated_decision_making']}")
    L.append(f"- **DPIA Conducted:** {inputs['dpia_conducted']}\n")

    L.append("## Privacy by Design — Principle Scores\n")
    L.append("| Principle | Score | Status |")
    L.append("|---|---|---|")
    for _, data in result["principle_scores"].items():
        L.append(f"| {data['name']} | {data['score']}/100 | {status_label(data['score'])} |")

    L.append("\n## Risk Register\n")
    if result["issues"]:
        L.append("| Principle | Severity | Finding | Recommendation |")
        L.append("|---|---|---|---|")
        for iss in sorted(result["issues"], key=lambda x: SEV_RANK[x["severity"]]):
            L.append(f"| {iss['principle']} | {iss['severity']} | {iss['finding']} | {iss['recommendation']} |")
    else:
        L.append("No issues identified.")

    L.append("\n## Applicable Regulations (Preliminary)\n")
    if regulations:
        for r in regulations:
            L.append(f"### {r['name']}")
            L.append(f"_Trigger: {r['reason']}_\n")
            for ob in r["obligations"]:
                L.append(f"- {ob}")
            L.append("")
    else:
        L.append("No specific regulatory triggers detected from current inputs.")

    L.append("\n---")
    L.append("_This report is generated by an automated Privacy by Design assessment tool for engineering "
              "triage purposes. It is decision support, not legal advice. Confirm all findings with your "
              "Legal / Privacy / DPO function before launch._")
    return "\n".join(L)


# ============================================================================
# 7. STREAMLIT APP
# ============================================================================

st.set_page_config(page_title="Privacy by Design Assessment Studio", page_icon="\U0001F6E1\uFE0F", layout="wide")

# --- Session state bootstrap -------------------------------------------------
for key, default in [
    ("history", []),
    ("result", None),
    ("data_elements_ms", []),
    ("req_text_input", ""),
    ("last_project_name", ""),
    ("last_inputs", None),
    ("last_regulations", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("\U0001F6E1\uFE0F Privacy by Design Assessment Studio")
st.caption(
    "A rule-based Privacy-by-Design (PbD) assessment tool for triaging engineering and product requirements. "
    "Runs entirely locally — nothing you type is sent to a third-party service. "
    "This is engineering-stage decision support, **not legal advice**."
)

# --- Sidebar: scoring weights (advanced, extensibility hook) ---------------
with st.sidebar:
    st.header("\U0001F4DD Requirement Intake")

    sample_choice = st.selectbox("Load a sample requirement (optional)", list(SAMPLE_REQUIREMENTS.keys()))
    if sample_choice != "-- Select a sample --" and SAMPLE_REQUIREMENTS[sample_choice] != st.session_state.req_text_input:
        st.session_state.req_text_input = SAMPLE_REQUIREMENTS[sample_choice]

    project_name = st.text_input("Project / Feature name", placeholder="e.g. Loyalty App Location Offers")

    requirement_text = st.text_area(
        "Describe the requirement in plain language",
        key="req_text_input",
        height=130,
        placeholder="What is being built? What data does it touch, and why?",
    )

    scan_col1, scan_col2 = st.columns([1, 1])
    with scan_col1:
        scan_clicked = st.button("\U0001F50D Scan text for data types", use_container_width=True)
    with scan_col2:
        reset_clicked = st.button("\U0001F504 Reset all", use_container_width=True)

    if reset_clicked:
        st.session_state.history = []
        st.session_state.result = None
        st.session_state.data_elements_ms = []
        st.session_state.req_text_input = ""
        st.rerun()

    if scan_clicked:
        suggestions = detect_pii_hints(requirement_text)
        merged = sorted(set(st.session_state.data_elements_ms) | set(suggestions))
        newly_added = sorted(set(suggestions) - set(st.session_state.data_elements_ms))
        st.session_state.data_elements_ms = merged
        if newly_added:
            st.success(f"Detected and added: {', '.join(newly_added)}")
        else:
            st.info("No new data types detected in the text.")

    st.divider()

    with st.expander("\u2696\uFE0F Principle weighting (advanced)", expanded=False):
        st.caption("Adjust if your organization weighs certain principles more heavily. Defaults are equal weight.")
        weights = {}
        for key, short, _, _ in PRINCIPLES:
            weights[key] = st.slider(short, 0.5, 2.0, 1.0, 0.1, key=f"w_{key}")

    with st.form("intake_form"):
        st.subheader("Processing Details")
        purpose = st.text_area("Business purpose of this processing", height=70,
                                placeholder="Why is this data needed? What business outcome does it serve?")
        data_elements = st.multiselect("Data elements involved", options=list(DATA_ELEMENTS.keys()), key="data_elements_ms")
        data_subjects = st.multiselect("Data subjects", options=SUBJECT_OPTIONS)
        geography = st.multiselect("Geography of data subjects", options=GEOGRAPHY_OPTIONS)

        st.subheader("Sharing, Retention & Consent")
        third_party_sharing = st.radio("Third-party sharing", options=SHARING_OPTIONS, horizontal=True)
        retention_defined = st.radio("Retention policy", options=RETENTION_OPTIONS)
        retention_period = st.text_input("If defined, specify the retention period", placeholder="e.g. 24 months post account closure")
        consent_mechanism = st.radio("Consent mechanism", options=CONSENT_OPTIONS)

        st.subheader("Security & Rights")
        security_measures = st.multiselect("Planned security measures", options=SECURITY_OPTIONS)
        dsr_support = st.multiselect("Data subject rights supported", options=DSR_OPTIONS)
        automated_decision_making = st.radio("Involves automated decision-making about individuals?", options=["No", "Yes"], horizontal=True)
        dpia_conducted = st.radio("Has a DPIA / PIA been conducted?", options=["No", "Yes", "Not sure"], horizontal=True)

        submitted = st.form_submit_button("\u25B6\uFE0F Run Privacy Assessment", use_container_width=True, type="primary")

# --- On submit: compute and store ------------------------------------------
if submitted:
    inputs = {
        "requirement_text": requirement_text,
        "purpose": purpose,
        "data_elements": data_elements,
        "data_subjects": data_subjects,
        "geography": geography,
        "third_party_sharing": third_party_sharing,
        "retention_defined": retention_defined,
        "retention_period": retention_period,
        "consent_mechanism": consent_mechanism,
        "security_measures": security_measures,
        "dsr_support": dsr_support,
        "automated_decision_making": automated_decision_making,
        "dpia_conducted": dpia_conducted,
    }
    result = run_assessment(inputs, weights)
    regulations = map_regulations(inputs)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    st.session_state.result = result
    st.session_state.last_inputs = inputs
    st.session_state.last_regulations = regulations
    st.session_state.last_project_name = project_name or "Untitled"

    high_count = sum(1 for i in result["issues"] if i["severity"] == "High")
    st.session_state.history.append({
        "Timestamp": timestamp,
        "Project": project_name or "Untitled",
        "Overall Score": result["overall_score"],
        "Status": status_label(result["overall_score"]),
        "High Severity Issues": high_count,
        "Total Issues": len(result["issues"]),
        "_full": {
            "inputs": inputs, "result": result, "regulations": regulations,
            "project_name": project_name or "Untitled", "timestamp": timestamp,
        },
    })

# --- Main area ---------------------------------------------------------------
if not st.session_state.result:
    st.info(
        "\U0001F448 Describe a requirement in the sidebar, optionally hit **Scan text for data types**, "
        "fill in the processing details, and click **Run Privacy Assessment** to see the results here.\n\n"
        "Try a sample requirement from the dropdown to see the tool in action."
    )
else:
    result = st.session_state.result
    inputs = st.session_state.last_inputs
    regulations = st.session_state.last_regulations
    project_name = st.session_state.last_project_name

    tab_overview, tab_deepdive, tab_risk, tab_reg, tab_export, tab_history = st.tabs(
        ["\U0001F4CA Overview", "\U0001F50E Principle Deep-Dive", "\u26A0\uFE0F Risk Register",
         "\u2696\uFE0F Regulatory Mapping", "\U0001F4C4 Export Report", "\U0001F553 History"]
    )

    # ---- Overview ----
    with tab_overview:
        c1, c2 = st.columns([1, 1.4])
        with c1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["overall_score"],
                title={"text": f"Overall Privacy Score — {project_name}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2b6cb0"},
                    "steps": [
                        {"range": [0, 50], "color": "#fde2e2"},
                        {"range": [50, 80], "color": "#fff3cd"},
                        {"range": [80, 100], "color": "#d4edda"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c2:
            cats = [data["name"] for data in result["principle_scores"].values()]
            vals = [data["score"] for data in result["principle_scores"].values()]
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself", name="Score"))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                                     showlegend=False, height=300, margin=dict(l=40, r=40, t=30, b=10))
            st.plotly_chart(fig_radar, use_container_width=True)

        high = [i for i in result["issues"] if i["severity"] == "High"]
        med = [i for i in result["issues"] if i["severity"] == "Medium"]
        low = [i for i in result["issues"] if i["severity"] == "Low"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Status", f"{status_emoji(result['overall_score'])} {status_label(result['overall_score'])}")
        m2.metric("High Severity", len(high))
        m3.metric("Medium Severity", len(med))
        m4.metric("Low Severity", len(low))

        if high:
            st.subheader("Top priorities before this ships")
            for iss in high[:5]:
                st.error(f"**[{iss['principle']}]** {iss['finding']}\n\n\u2192 {iss['recommendation']}")
        else:
            st.success("No high-severity issues identified against the current inputs.")

    # ---- Principle Deep-Dive ----
    with tab_deepdive:
        st.caption("Each of the 7 Foundational Principles of Privacy by Design, scored independently and transparently.")
        for key, short, desc, _ in PRINCIPLES:
            data = result["principle_scores"][key]
            related = [i for i in result["issues"] if i["principle"] == short]
            with st.expander(f"{status_emoji(data['score'])} {short} — {data['score']}/100 ({status_label(data['score'])})", expanded=False):
                st.write(desc)
                if related:
                    for iss in related:
                        badge = {"High": "\U0001F534", "Medium": "\U0001F7E1", "Low": "\U0001F535"}[iss["severity"]]
                        st.markdown(f"{badge} **{iss['severity']}** — {iss['finding']}")
                        st.caption(f"Recommendation: {iss['recommendation']}")
                else:
                    st.write("No issues identified for this principle.")

    # ---- Risk Register ----
    with tab_risk:
        if result["issues"]:
            df = pd.DataFrame(result["issues"])[["principle", "severity", "finding", "recommendation"]]
            df.columns = ["Principle", "Severity", "Finding", "Recommendation"]
            df["_rank"] = df["Severity"].map(SEV_RANK)
            df = df.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)

            sev_filter = st.multiselect("Filter by severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
            filtered = df[df["Severity"].isin(sev_filter)]

            def color_severity(val):
                colors = {"High": "background-color: #fde2e2", "Medium": "background-color: #fff3cd", "Low": "background-color: #e2eefd"}
                return colors.get(val, "")

            st.dataframe(filtered.style.applymap(color_severity, subset=["Severity"]), use_container_width=True, height=420)
            st.download_button("\u2B07\uFE0F Download risk register (CSV)", data=df.to_csv(index=False),
                                file_name=f"{project_name}_risk_register.csv", mime="text/csv")
        else:
            st.success("No risks identified for the current requirement inputs.")

    # ---- Regulatory Mapping ----
    with tab_reg:
        st.caption("Preliminary, heuristic mapping based on data types and geography. Always confirm with Legal.")
        if regulations:
            for r in regulations:
                with st.expander(r["name"], expanded=False):
                    st.write(f"**Trigger:** {r['reason']}")
                    for ob in r["obligations"]:
                        st.markdown(f"- {ob}")
        else:
            st.info("No specific regulatory triggers detected from current inputs. General best practice still applies.")

    # ---- Export ----
    with tab_export:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        md_report = generate_markdown_report(project_name, timestamp, inputs, result, regulations)
        json_report = json.dumps({
            "project_name": project_name, "timestamp": timestamp, "inputs": inputs,
            "overall_score": result["overall_score"], "principle_scores": result["principle_scores"],
            "issues": result["issues"], "regulations": regulations,
        }, indent=2)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("\u2B07\uFE0F Download report (Markdown)", data=md_report,
                                file_name=f"{project_name}_pbd_report.md", mime="text/markdown", use_container_width=True)
        with col2:
            st.download_button("\u2B07\uFE0F Download machine-readable (JSON)", data=json_report,
                                file_name=f"{project_name}_pbd_report.json", mime="application/json", use_container_width=True)

        st.subheader("Report preview")
        st.markdown(md_report)

    # ---- History ----
    with tab_history:
        if st.session_state.history:
            hist_df = pd.DataFrame([{k: v for k, v in h.items() if k != "_full"} for h in st.session_state.history])
            st.dataframe(hist_df, use_container_width=True)
            st.download_button("\u2B07\uFE0F Download history (CSV)", data=hist_df.to_csv(index=False),
                                file_name="pbd_assessment_history.csv", mime="text/csv")
            options = [f"{i+1}. {h['Project']} ({h['Timestamp']})" for i, h in enumerate(st.session_state.history)]
            pick = st.selectbox("View a past assessment", options)
            idx = options.index(pick)
            full = st.session_state.history[idx]["_full"]
            with st.expander("Full report for selected assessment", expanded=True):
                st.markdown(generate_markdown_report(full["project_name"], full["timestamp"], full["inputs"], full["result"], full["regulations"]))
        else:
            st.info("No assessments run yet this session. History resets when the app restarts (see notes in the write-up for persistence options).")