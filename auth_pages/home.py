import os
import streamlit as st
import pandas as pd
from datetime import datetime

# Paths - Keeping your existing structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(BASE_DIR, "final_dataset.csv")
RISK_HISTORY_DIR = os.path.join(BASE_DIR, "risk_history")
LAB_CSV = os.path.join(BASE_DIR, "user_lab_reports.csv")


@st.cache_data(show_spinner=False)
def _get_dashboard_stats():
    """Fetches global platform statistics."""
    try:
        df = pd.read_csv(DATASET)
        return {
            "records": len(df),
            "diseases": df["Disease"].nunique() if "Disease" in df.columns else 41,
            "states": df["State"].nunique() if "State" in df.columns else 28,
            "avg_risk": round(df["Risk_Score"].mean(), 1) if "Risk_Score" in df.columns else 42.3
        }
    except Exception:
        return {"records": 30000, "diseases": 41, "states": 28, "avg_risk": 42.3}


def _get_user_snapshot(uid):
    """Fetches the specific user's latest risk data."""
    path = os.path.join(RISK_HISTORY_DIR, f"{uid}.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if not df.empty:
                latest = df.sort_values("timestamp", ascending=False).iloc[0]
                return {
                    "score": float(latest.get("score", 0)),
                    "label": str(latest.get("risk_label", "Low")),
                    "count": len(df)
                }
        except Exception:
            pass
    return None


def page_home():
    # 1. Setup & Context
    uid = st.session_state.get("uid", "guest")
    name = st.session_state.get("username", "User")
    first_name = name.split()[0].capitalize()

    now = datetime.now()
    greeting = "Good morning" if now.hour < 12 else ("Good afternoon" if now.hour < 17 else "Good evening")

    # 2. Hero Section
    st.markdown(f"""
        <div class="hero">
            <div class="hero-greet">{greeting}</div>
            <div class="hero-name">{first_name}</div>
            <div class="hero-date">{now.strftime("%A, %B %d, %Y")}</div>
            <div class="hero-tag">
                Welcome back to your health intelligence command center. 
                Monitor population trends and personal risk analytics in real-time.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. KPI Row (Matches your grid-template-columns: repeat(4, 1fr))
    stats = _get_dashboard_stats()
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi">
            <div class="kpi-icon">🗄️</div>
            <div class="kpi-val">{stats['records']:,}</div>
            <div class="kpi-lbl">Training Records</div>
            <div class="kpi-sub">Global Dataset</div>
        </div>
        <div class="kpi">
            <div class="kpi-icon">🦠</div>
            <div class="kpi-val">{stats['diseases']}</div>
            <div class="kpi-lbl">Diseases Tracked</div>
            <div class="kpi-sub">Medical Categories</div>
        </div>
        <div class="kpi">
            <div class="kpi-icon">📍</div>
            <div class="kpi-val">{stats['states']}</div>
            <div class="kpi-lbl">States Covered</div>
            <div class="kpi-sub">Pan-India Analytics</div>
        </div>
        <div class="kpi">
            <div class="kpi-icon">📊</div>
            <div class="kpi-val">{stats['avg_risk']}%</div>
            <div class="kpi-lbl">Population Risk</div>
            <div class="kpi-sub">Baseline Average</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Personal Risk Panel
    user_data = _get_user_snapshot(uid)

    if user_data:
        # Dynamic styling based on risk level
        is_low = user_data['label'].lower() == "low"
        css_class = "panel-low" if is_low else "panel-med"
        score_class = "score-low" if is_low else "score-med"
        badge_class = "badge-low" if is_low else "badge-med"
        hint = "Maintain your healthy lifestyle." if is_low else "Review your personalized health plan."

        st.markdown(f"""
        <div class="panel {css_class}">
            <div class="panel-lbl">Your Personalized Risk Score</div>
            <div class="score-big {score_class}">{user_data['score']}</div>
            <div><span class="badge {badge_class}">{user_data['label']} Risk</span></div>
            <div class="panel-hint">
                {hint} <br/>
                <small>{user_data['count']} historical assessments tracked</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="panel panel-green">
            <div class="panel-lbl">Your Risk Score</div>
            <div class="score-big score-none">—</div>
            <div class="panel-hint">
                No data available. Complete a <b>Risk Assessment</b> to see your score here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 5. Feature Discovery Grid
    st.markdown('<div class="sec-title">Core Capabilities</div>', unsafe_allow_html=True)

    features = [
        ("", "ML Prediction", "Multi-algorithm disease forecasting."),
        ("", "Risk Profiling", "Dynamic health scoring & history."),
        ("", "Lab Insights", "AI-driven biomarker extraction."),
        ("", "Health Bot", "LLM-powered medical assistant."),
        ("", "Geo-Tracker", "Regional outbreak visualization."),
        ("", "Reports", "Export full health history to PDF.")
    ]

    cols_html = "".join([f"""
        <div class="feat">
            <div class="feat-icon">{icon}</div>
            <div class="feat-name">{name}</div>
            <div class="feat-desc">{desc}</div>
        </div>""" for icon, name, desc in features])

    st.markdown(f'<div class="feat-grid">{cols_html}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    page_home()