import streamlit as st
import pandas as pd
import os


USER_DB = "user_registration_data.csv"


def page_otp_verify():
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("<h1 style='text-align: center;'>Enter OTP</h1>", unsafe_allow_html=True)

        # Determine which email to show (current email or the one stored in session)
        display_email = st.session_state.get('otp_email', st.session_state.get('user_email', 'your email'))

        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 10px; margin-top: 10px;">
                We sent a 6-digit OTP to <b>{display_email}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("otp_form"):
            entered = st.text_input("Enter OTP", placeholder="Enter 6-digit code", max_chars=6)
            submit = st.form_submit_button("Verify & Continue", use_container_width=True)

        if submit:
            if not entered:
                st.error("Please enter the OTP.")
            elif entered.strip() != st.session_state.get("otp", ""):
                st.error("Incorrect OTP. Please try again.")
            else:
                # --- OTP VERIFIED SUCCESSFULLY ---

                # 1. Handle Password Update Case
                if "pending_new_pw" in st.session_state:
                    new_pw = st.session_state.get("pending_new_pw")
                    df = pd.read_csv(USER_DB)
                    df.loc[df['email'] == st.session_state.user_email, 'password'] = new_pw
                    df.to_csv(USER_DB, index=False)
                    st.session_state.pop("pending_new_pw")
                    st.success("✅ Password updated successfully!")
                    st.session_state.page = "welcome"  # Return to dashboard/settings

                # 2. Handle Email Update Case
                elif "pending_new_email" in st.session_state:
                    new_email = st.session_state.get("pending_new_email")
                    df = pd.read_csv(USER_DB)
                    # Update email in the CSV record
                    df.loc[df['email'] == st.session_state.user_email, 'email'] = new_email
                    df.to_csv(USER_DB, index=False)

                    st.session_state.user_email = new_email
                    st.session_state.uid = new_email  # Update UID if linked to email
                    st.session_state.pop("pending_new_email")
                    st.success("✅ Email updated successfully!")
                    st.session_state.page = "welcome"

                # 3. Handle Standard Login Case
                else:
                    st.query_params["uid"] = st.session_state.get("uid", "")
                    st.query_params["email"] = st.session_state.get("user_email", "")
                    st.query_params["username"] = st.session_state.get("username", "")

                    if st.session_state.get("is_new_user", True):
                        st.success("Verified! Let's set up your health profile.")
                        st.session_state.page = "questionnaire"
                    else:
                        st.success("Verified! Welcome back.")
                        st.session_state.page = "welcome"

                # Cleanup OTP session data
                st.session_state.pop("otp", None)
                st.rerun()

        # Footer Buttons
        if st.button("Resend OTP", use_container_width=True):
            new_otp = generate_otp()
            st.session_state["otp"] = new_otp
            success, error = send_otp(display_email, new_otp)
            if success:
                st.success("OTP resent!")
            else:
                st.error(f"Error: {error}")

        if st.button("Cancel", use_container_width=True):
            # Clean up pending changes if user cancels
            st.session_state.pop("pending_new_pw", None)
            st.session_state.pop("pending_new_email", None)
            st.session_state.page = "login" if "user_email" not in st.session_state else "welcome"
            st.rerun()