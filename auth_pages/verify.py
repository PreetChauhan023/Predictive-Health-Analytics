import time
import streamlit as st
from firebase_config import auth, admin_auth


def page_verify():
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("Check Your Email ✉️")
        st.write(f"A verification link was sent to **{st.session_state.user_email}**")

        st.info("️ Click the verification link in your email. This page will redirect automatically once verified.")

        # ── Auto polling — checks every 3 seconds ──
        try:
            user_record = admin_auth.get_user_by_email(st.session_state.user_email)
            if user_record.email_verified:
                st.session_state.username = user_record.display_name or st.session_state.user_email
                st.session_state.page     = "login"
                st.success(" Email verified! Redirecting to login...")
                st.rerun()
        except Exception:
            pass

        with st.spinner("Waiting for email verification..."):
            time.sleep(3)
            st.rerun()

        # ── Manual controls as fallback ──
        with st.form("verify_form"):
            resend = st.form_submit_button("Resend Verification Email", use_container_width=True)
            back   = st.form_submit_button("Back to Login", use_container_width=True)

        if resend:
            try:
                auth.send_email_verification(st.session_state.id_token)
                st.success("✅ Verification email resent! Check your inbox.")
            except Exception as e:
                st.error(f"❌ Could not resend: {e}")

        if back:
            st.session_state.page = "login"
            st.rerun()