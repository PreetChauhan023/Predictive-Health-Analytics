import streamlit as st
import os
import pandas as pd
import hashlib
from datetime import date
from utils.validators import is_valid_email, is_valid_phone, is_strong_password


def hash_password(password: str) -> str:
    """Return SHA-256 hex digest of the password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

USER_DB = "user_registration_data.csv"

def calculate_age(dob):
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

def init_user_db():
    if not os.path.exists(USER_DB):
        df = pd.DataFrame(columns=[
            "first_name", "last_name", "email", "phone",
            "dob", "age", "height_cm", "weight_kg", "blood_group",
            "username", "password", "created_at"
        ])
        df.to_csv(USER_DB, index=False)

def page_signup():
    init_user_db()

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("<h2 style='text-align: center;'>Create Account</h2>", unsafe_allow_html=True)


        with st.form("signup_form"):
            c1, c2 = st.columns(2)
            with c1:
                first_name = st.text_input("First Name", placeholder="John")
            with c2:
                last_name = st.text_input("Last Name", placeholder="Doe")

            c3, c4 = st.columns(2)
            with c3:
                email = st.text_input("Email", placeholder="you@example.com")
            with c4:
                phone = st.text_input("Phone", placeholder="+91XXXXXXXXXX")

            dob = st.date_input(
                "Date of Birth",
                min_value=date(1900, 1, 1),
                max_value=date.today(),
                value=date(2000, 1, 1)
            )

            # NEW: Auto age
            age = calculate_age(dob)

            # NEW: Health fields
            c5, c6 = st.columns(2)
            with c5:
                height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
            with c6:
                weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)

            blood_group = st.selectbox(
                "Blood Group",
                ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
            )

            username = st.text_input("Username", placeholder="johndoe123")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            confirm = st.text_input("Re-enter Password", type="password", placeholder="••••••••")

            terms = st.checkbox(
                "By signing up, you agree to our **Terms of Service and Privacy Policy**."
            )

            submit = st.form_submit_button("Create Account", use_container_width=True)

        if st.button("Already have an account? Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

        if submit:
            if not all([first_name, last_name, email, phone, username, password, confirm]):
                st.error("All fields are required.")
            elif not is_valid_email(email):
                st.error("Invalid email format.")
            elif not is_valid_phone(phone):
                st.error("Invalid Indian phone number.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif not terms:
                st.error("You must agree to our Terms of Service.")
            else:
                pw_error = is_strong_password(password)
                if pw_error:
                    st.error(f"{pw_error}")
                else:
                    try:
                        df = pd.read_csv(USER_DB)

                        if email in df['email'].values:
                            st.error("❌ Email already registered.")
                        elif username in df['username'].values:
                            st.error("❌ Username already taken.")
                        else:
                            new_user = {
                                "first_name": first_name,
                                "last_name": last_name,
                                "email": email,
                                "phone": phone,
                                "dob": str(dob),
                                "age": age,
                                "height_cm": height,
                                "weight_kg": weight,
                                "blood_group": blood_group,
                                "username": username,
                                "password": hash_password(password),
                                "created_at": date.today()
                            }

                            df = pd.concat([df, pd.DataFrame([new_user])], ignore_index=True)
                            df.to_csv(USER_DB, index=False)

                            st.session_state.page = "login"
                            st.success("✅ Account created! Please login with your credentials.")
                            st.rerun()

                    except Exception as e:
                        st.error(f"❌ Signup failed: {str(e)}")