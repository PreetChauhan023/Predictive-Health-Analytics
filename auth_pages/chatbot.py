import os
import pandas as pd
import streamlit as st
from groq import Groq
from datetime import datetime
from utils.config import GROQ_API_KEY

# Configuration and Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB_CSV = os.path.join(BASE_DIR, "data", "user_lab_reports.csv")
CHAT_CSV = os.path.join(BASE_DIR, "chat_history.csv")
POPULATION_CSV = os.path.join(BASE_DIR, "../final_dataset.csv")
REG_CSV = os.path.join(BASE_DIR, "user_registration_data.csv")

client = Groq(api_key=GROQ_API_KEY)

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

# ── Data Helpers ─────────────────────────────────────

def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()


def get_user_rows(path, uid):
    df = read_csv_safe(path)
    if df.empty:
        return df, pd.DataFrame()

    for col in ["uid", "email", "username"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            user_rows = df[df[col] == uid]
            if not user_rows.empty:
                return df, user_rows

    return df, pd.DataFrame()


@st.cache_data
def load_population_summary():
    if not os.path.exists(POPULATION_CSV):
        return ""

    df = read_csv_safe(POPULATION_CSV)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    def avg(col):
        return round(df[col].mean(), 1) if col in df.columns else "N/A"

    top_diseases = ""
    if "disease_reported" in df.columns:
        top_diseases = ", ".join(
            f"{d} ({c})"
            for d, c in df[df["disease_reported"] != "Healthy"]["disease_reported"]
            .value_counts().head(5).items()
        )

    return (
        f"POPULATION BENCHMARK:\n"
        f"- Avg Age: {avg('age')} | Avg BMI: {avg('bmi')} | Avg Risk: {avg('risk_score')}/100\n"
        f"- Top Diseases: {top_diseases}"
    )


def build_registration_context(uid):
    _, user_rows = get_user_rows(REG_CSV, uid)
    if user_rows.empty:
        return ""

    row = user_rows.iloc[0]

    return "USER PROFILE INFO:\n" + "\n".join([
        f"- Age: {row.get('age', 'N/A')}",
        f"- Height: {row.get('height_cm', 'N/A')} cm",
        f"- Weight: {row.get('weight_kg', 'N/A')} kg",
        f"- Blood Group: {row.get('blood_group', 'N/A')}",
    ])


def build_lab_context(uid):
    df, user_rows = get_user_rows(LAB_CSV, uid)
    if user_rows.empty:
        return ""

    lines = []
    for label, col in FLAT_LAB_COLUMNS.items():
        if col in df.columns:
            values = user_rows[col].dropna()
            if not values.empty:
                lines.append(f"- {label}: {values.iloc[0]}")

    latest_row = user_rows.iloc[-1]
    summary = [
        f"- Interpretation: {latest_row.get('overall_interpretation')}",
        f"- Advice: {latest_row.get('advice')}"
    ]

    return "USER LAB REPORT:\n" + "\n".join(summary + lines)


# ── Chat Persistence ─────────────────────────────────

def save_message(role, content):
    pd.DataFrame([{
        "uid": st.session_state.get("uid"),
        "email": st.session_state.get("user_email"),  # ✅ FIXED
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }]).to_csv(CHAT_CSV, mode="a", index=False, header=not os.path.exists(CHAT_CSV))


def load_chat_history():
    uid = str(st.session_state.get("uid", "")).strip()
    if not os.path.exists(CHAT_CSV):
        return []

    df = read_csv_safe(CHAT_CSV)
    if df.empty:
        return []

    match_col = "uid" if "uid" in df.columns else ("email" if "email" in df.columns else None)
    if not match_col:
        return []

    df[match_col] = df[match_col].astype(str).str.strip()
    df["role"] = df["role"].astype(str).str.strip().str.lower()

    # KEEP ONLY VALID ROLES
    valid_roles = ["user", "assistant", "system"]
    df = df[df["role"].isin(valid_roles)]

    return df[df[match_col] == uid][["role", "content"]].to_dict("records")


def is_personal_query(query):
    query = query.lower()

    personal_keywords = [
        "my", "mine", "me", "i", "my report", "my health",
        "my cholesterol", "my sugar", "my blood",
        "am i", "do i have", "my results"
    ]

    return any(word in query for word in personal_keywords)


# ── Main UI ──────────────────────────────────────────

def page_chatbot():
    st.title("Health AI Assistant")
    if not st.session_state.get("uid"):
        st.warning("Please log in first.")
        return
    uid = str(st.session_state.get("uid", "")).strip()
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    # Input
    if user_input := st.chat_input("Ask about your health..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        save_message("user", user_input)
        if is_personal_query(user_input):
            user_context = "\n\n".join(filter(None, [
                build_registration_context(uid),
                build_lab_context(uid)
            ]))
        else:
            user_context = load_population_summary()

        system_prompt = f"""
        You are a Health AI Assistant.

        RULES:
        - If user asks about THEIR health → use USER CONTEXT
        - If user asks GENERAL or DATASET questions → IGNORE personal data
        - Never mix personal lab report with general answers

        CONTEXT:
        {user_context}

        Be clear, accurate, and professional.
        """

        with st.chat_message("assistant"):
            try:
                with st.spinner("Processing..."):

                    # ✅ CLEAN MESSAGES BEFORE SENDING
                    valid_roles = ["user", "assistant", "system"]

                    clean_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[-6:]
                        if m.get("role") in valid_roles and isinstance(m.get("content"), str)
                    ]

                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": system_prompt}] + clean_messages,
                        temperature=0.3,
                    )

                reply = response.choices[0].message.content
                st.write(reply)

                st.session_state.messages.append({"role": "assistant", "content": reply})
                save_message("assistant", reply)

            except Exception as e:
                st.error(f"Chat Error: {e}")