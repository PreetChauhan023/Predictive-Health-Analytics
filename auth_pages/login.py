import streamlit as st
import pandas as pd
import os
import hashlib
from utils.validators import is_valid_email

USER_DB = "user_registration_data.csv"


def hash_password(password: str) -> str:
    """Return SHA-256 hex digest of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _check_password(input_pw: str, stored_pw: str) -> bool:
    """
    Accept both legacy plain-text passwords and new hashed ones.
    Once the user logs in via the plain-text path, their record is NOT
    auto-upgraded here — that would require a write on every login.
    New sign-ups always store a hash, so the plain-text path will
    disappear naturally as users re-register.
    """
    hashed = hash_password(input_pw)
    if stored_pw == hashed:
        return True
    # Fallback: legacy plain-text (allows existing users to still log in)
    if stored_pw == input_pw:
        return True
    return False

def page_login():
    _, center, _ = st.columns([1, 2, 1])

    with center:

        st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
        #st.markdown("<p style='text-align: center;'>Sign in to your health dashboard</p>", unsafe_allow_html=True)

        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", placeholder="••••••••", type="password")
            submit = st.form_submit_button("Access Platform", use_container_width=True)

        if st.button("Don't have an account? Register here", use_container_width=True):
            st.session_state.page = "signup"
            st.rerun()

        if submit:
            if not email or not password:
                st.error("Please fill in all fields.")

            elif not is_valid_email(email):
                st.error("Enter a valid email address.")

            elif not os.path.exists(USER_DB):
                st.error("No user database found. Please sign up first.")

            else:
                try:
                    df = pd.read_csv(USER_DB)
                    user_row = df[df['email'] == email]

                    if user_row.empty:
                        st.error("No account found with this email.")

                    elif not _check_password(password, str(user_row.iloc[0]['password'])):
                        st.error("Incorrect password.")

                    else:
                        user_data = user_row.iloc[0]

                        # ── Store user session data ──
                        st.session_state.user_email    = email
                        st.session_state.user_username = user_data.get("username", "N/A")
                        st.session_state.username      = f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip()
                        st.session_state.user_phone    = user_data.get("phone", "N/A")
                        st.session_state.user_dob      = user_data.get("dob", "N/A")
                        st.session_state.uid           = email


                        # ── Skip OTP & go directly to dashboard ──
                        st.query_params["uid"]      = email
                        st.query_params["email"]    = email
                        st.query_params["username"] = st.session_state.username

                        # ✅ ALWAYS go to dashboard
                        st.session_state.page = "welcome"

                        st.rerun()

                except Exception as e:
                    st.error(f"Login failed: {str(e)}")