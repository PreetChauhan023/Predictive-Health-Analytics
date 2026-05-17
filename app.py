import streamlit as st
import pandas as pd
import os

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predictive Health Analytics", layout="wide")

# ─── LOAD CSS ──────────────────────────────────────────────────────────────────
if os.path.exists("style.css"):
    with open("style.css", encoding="utf-8") as f:
      st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
for key, val in {
    "page":            "login",
    "user_email":      "",
    "id_token":        "CSV_MODE",
    "username":        "",
    "uid":             "",
    "user_username":   "",
    "user_phone":      "",
    "user_dob":        "",
    "user_gender":     "",
    "is_new_user":     False,
    "authenticated":   False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─── QUERY PARAM SESSION RESTORE (survives hard refresh) ───────────────────────
def _restore_session_from_query_params():
    """If user hard-refreshes, restore session from URL query params."""
    params = st.query_params
    uid = params.get("uid", "")
    if not uid or st.session_state.get("uid"):
        return  # Already restored or nothing to restore

    USER_DB = "user_registration_data.csv"
    HEALTH_CSV = "user_health_responses.csv"

    if not os.path.exists(USER_DB):
        return

    try:
        df = pd.read_csv(USER_DB, dtype=str)
        # uid == email in CSV mode
        user_row = df[df["email"] == uid]
        if user_row.empty:
            return

        user = user_row.iloc[0]
        st.session_state.uid           = uid
        st.session_state.user_email    = uid
        st.session_state.username      = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
        st.session_state.user_username = user.get("username", "")
        st.session_state.user_phone    = user.get("phone", "")
        st.session_state.user_dob      = user.get("dob", "")
        st.session_state.page = "welcome"
    except Exception:
        pass
_restore_session_from_query_params()

# ─── ROUTER ────────────────────────────────────────────────────────────────────
from auth_pages.login         import page_login
from auth_pages.signup        import page_signup
from auth_pages.otp_verify    import page_otp_verify
from auth_pages.dashboard     import page_dashboard

if st.session_state.page == "login":
    page_login()
elif st.session_state.page == "signup":
    page_signup()
elif st.session_state.page == "welcome":
    page_dashboard()