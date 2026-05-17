import os
import streamlit as st
import pandas as pd
import numpy as np
from utils.config import GROQ_API_KEY, GEMINI_API_KEY
from services.interpreter import LabInterpreter
from storage.repository import LabRepository
from utils.visualization import render_donut

# ── Paths & Config ───────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB_CSV = os.path.join(BASE_DIR, "data", "user_lab_reports.csv")
REG_CSV = os.path.join(BASE_DIR, "data", "user_registration_data.csv")
os.makedirs(os.path.dirname(LAB_CSV), exist_ok=True)

# ── Services ─────────────────────────────────────────
interpreter = LabInterpreter(api_key=GROQ_API_KEY, gemini_api_key=GEMINI_API_KEY)
repository = LabRepository(file_path=LAB_CSV)

FLAT_LAB_COLUMNS = {
    "Hemoglobin": "cbc_hemoglobin",
    "Cholesterol": "lipid_profile_cholesterol",
    "Triglycerides": "lipid_profile_triglyceride",
    "HDL Cholesterol": "lipid_profile_hdl_cholesterol",
    "LDL Cholesterol": "lipid_profile_direct_ldl",
    "VLDL": "lipid_profile_vldl",
    "Fasting Blood Sugar": "blood_sugar_fasting_blood_sugar",
    "HbA1c": "blood_sugar_hba1c",
    "ESR": "other_esr",
    "RBC Count": "cbc_rbc_count",
    "WBC Count": "cbc_wbc_count",
    "Platelet Count": "cbc_platelet_count",
}


# ── Helpers ──────────────────────────────────────────

def safe_int(val, default=0):
    if pd.isna(val) or val is None:
        return default
    try:
        return int(float(val))
    except:
        return default


def get_user_profile(uid):
    if not os.path.exists(REG_CSV):
        return None
    try:
        df = pd.read_csv(REG_CSV)
        user_row = df[(df['uid'].astype(str) == str(uid)) | (df['username'].astype(str) == str(uid))]
        return user_row.iloc[0] if not user_row.empty else None
    except:
        return None


def _status_badge(val):
    color = {
        "High": "#ff4b4b",
        "Low": "#ff4b4b",
        "Borderline": "#ffa500",
        "Normal": "#00c853"
    }.get(val, "#808080")
    return f"background-color: {color}; color: white; padding: 2px 5px; border-radius: 4px;"


# ── Page Logic ───────────────────────────────────────

def page_labreport():
    st.title("Lab Report Analytic Center")
    uid = st.session_state.get("uid")
    email = st.session_state.get("user_email")
    if not uid:
        st.warning("Please log in to access your health analytics.")
        return
    tab_analysis, tab_history = st.tabs(["Upload report", "Report History"])
    with tab_analysis:
        profile = get_user_profile(uid)
        if profile is not None:
            with st.container(border=True):
                st.markdown("##### Current Health Profile")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Age", f"{profile.get('age', 'N/A')} yrs")
                c2.metric("Blood", profile.get('blood_group', 'N/A'))

                try:
                    height_m = float(profile['height_cm']) / 100
                    weight_kg = float(profile['weight_kg'])
                    bmi = round(weight_kg / (height_m ** 2), 1)
                    c3.metric("BMI", bmi)
                except:
                    c3.metric("BMI", "N/A")

                c4.metric("Weight", f"{profile.get('weight_kg', 'N/A')} kg")
        st.divider()
        st.subheader("Analyze New Report")
        uploaded_file = st.file_uploader("Upload Lab Report (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"])
        if uploaded_file:
            if st.button("Analyze Report ", use_container_width=True):
                with st.spinner("AI is interpreting biomarkers based on your profile..."):
                    file_bytes = uploaded_file.read()
                    result = interpreter.interpret(
                        file_bytes,
                        uploaded_file.name,
                        user_profile=profile.to_dict() if profile is not None else None
                    )
                    if result:
                        repository.save_report(uid, email, result)
                        # ✅ STORE RESULT
                        st.session_state["latest_report"] = result
                        st.success("Analysis complete!")
                    else:
                        st.error("Failed to interpret the report.")

        # ─────────────────────────────────────────────
        # ✅ SHOW RESULT IN SAME TAB
        # ─────────────────────────────────────────────
        latest = st.session_state.get("latest_report")

        if latest:
            st.divider()
            st.header("Analysis Result")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Executive Summary")
                st.write(latest.get("overall_interpretation", ""))

                st.markdown("### Critical Flags")
                for flag in latest.get("critical_flags", []):
                    st.error(flag)

                st.markdown("### Possible Conditions")
                for cond in latest.get("possible_conditions", []):
                    st.warning(cond)

                st.markdown("### Advice")
                st.info(latest.get("advice", ""))

            with col2:
                counts = latest.get("counts", {})

                if counts:
                    render_donut({
                        "Normal": counts.get("normal", 0),
                        "High": counts.get("high", 0),
                        "Low": counts.get("low", 0),
                        "Borderline": counts.get("borderline", 0)
                    })

            # 📊 CATEGORY CHART
            st.markdown("### Category Distribution")

            categories = latest.get("categories", {})
            cat_counts = {k: len(v) for k, v in categories.items() if len(v) > 0}

            if cat_counts:
                chart_df = pd.DataFrame({
                    "Category": list(cat_counts.keys()),
                    "Tests": list(cat_counts.values())
                })

                st.bar_chart(chart_df.set_index("Category"))

            # 🧪 BIOMARKER TABLE
            st.markdown("### Biomarker Details")

            records = []
            for cat, tests in latest.get("categories", {}).items():
                for t in tests:
                    records.append({
                        "Category": cat,
                        "Test": t.get("name"),
                        "Value": f"{t.get('value')} {t.get('unit')}",
                        "Status": t.get("status"),
                        "Meaning": t.get("meaning")
                    })

            if records:
                df = pd.DataFrame(records)
                st.dataframe(
                    df.style.map(_status_badge, subset=["Status"]),
                    use_container_width=True,
                    hide_index=True
                )

    # ─────────────────────────────────────────────
    # TAB 2: HISTORY (UNCHANGED)
    # ─────────────────────────────────────────────
    with tab_history:
        if not os.path.exists(LAB_CSV):
            st.info("No reports saved yet.")
        else:
            df_all = pd.read_csv(LAB_CSV)
            df_user = df_all[df_all["uid"].astype(str) == str(uid)].sort_index(ascending=False)

            if df_user.empty:
                st.info("No historical reports found.")
            else:
                for idx, row in df_user.iterrows():
                    is_expanded = (idx == df_user.index[0])

                    with st.expander(f"Report Date: {row.get('analyzed_at', 'N/A')}", expanded=is_expanded):

                        h_col1, h_col2 = st.columns([2, 1])

                        with h_col1:
                            st.markdown("### Executive Summary")
                            st.write(row.get("overall_interpretation", ""))

                            st.markdown("### 💡 Advice")
                            st.info(row.get("advice", ""))

                            # ✅ ADD BIOMARKER DETAILS
                            st.markdown("### 🧪 Biomarker Details")

                            records = []
                            for label, col in FLAT_LAB_COLUMNS.items():
                                val = row.get(f"{col}_value")
                                status = row.get(f"{col}_status")

                                if pd.notna(val):
                                    records.append({
                                        "Test": label,
                                        "Value": val,
                                        "Status": status
                                    })
                            if records:
                                hist_df = pd.DataFrame(records)
                                st.dataframe(
                                    hist_df.style.map(_status_badge, subset=["Status"]),
                                    use_container_width=True,
                                    hide_index=True
                                )

                        with h_col2:
                            counts = {
                                "Normal": safe_int(row.get("normal")),
                                "High": safe_int(row.get("high")),
                                "Low": safe_int(row.get("low")),
                                "Borderline": safe_int(row.get("borderline"))
                            }

                            if sum(counts.values()) > 0:
                                render_donut(counts)
if __name__ == "__main__":
    page_labreport()