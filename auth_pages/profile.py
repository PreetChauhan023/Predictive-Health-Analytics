import os
import streamlit as st
import pandas as pd
from datetime import date

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use registration DB instead of health responses
USER_DB = os.path.join(BASE_DIR, "user_registration_data.csv")

def page_profile():

    st.title("My Profile")
    st.write("Your personal details and health snapshot.")
    st.divider()

    # ── Session data ──────────────────────────────────────
    email = st.session_state.get("user_email", "N/A")

    # Default values
    name = username = phone = dob = "N/A"
    age = height = weight = bmi = blood_group = "N/A"

    # ── Load data from USER REGISTRATION CSV ──────────────
    if email != "N/A" and os.path.exists(USER_DB):
        try:
            df = pd.read_csv(USER_DB)

            if "email" in df.columns and email in df["email"].values:
                row = df[df["email"] == email].iloc[0]

                # Personal Info
                name = f"{row.get('first_name', '')} {row.get('last_name', '')}".strip()
                username = row.get("username", "N/A")
                phone = row.get("phone", "N/A")
                dob = row.get("dob", "N/A")

                # Health Info
                age = row.get("age", "N/A")
                height = row.get("height_cm", "N/A")
                weight = row.get("weight_kg", "N/A")
                blood_group = row.get("blood_group", "N/A")

                # Format values
                if height not in ["", "N/A"]:
                    height = f"{float(height):.0f}"
                if weight not in ["", "N/A"]:
                    weight = f"{float(weight):.0f}"

                # BMI calculation
                if height not in ["N/A"] and weight not in ["N/A"]:
                    try:
                        bmi_val = float(weight) / (float(height) / 100) ** 2
                        bmi = f"{bmi_val:.1f}"
                    except:
                        bmi = "N/A"

                # Age fallback (if missing in CSV)
                if age in ["", "N/A"] and dob not in ["", "N/A"]:
                    try:
                        birth = date.fromisoformat(str(dob))
                        today = date.today()
                        age = str(today.year - birth.year - (
                            (today.month, today.day) < (birth.month, birth.day)
                        ))
                    except:
                        age = "N/A"

        except Exception as e:
            st.error(f"Error loading profile: {str(e)}")

    # ── Personal Info Card ────────────────────────────────
    with st.container(border=True):
        col_left, _ = st.columns([2, 3])
        with col_left:
            st.subheader(name if name.strip() else "N/A")
            st.caption(f"Email: {email}")

        st.divider()

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Username", username)
        with c2: st.metric("Phone", phone)
        with c3: st.metric("Date of Birth", dob)

    st.write("")

    # ── Health Snapshot Card ──────────────────────────────
    with st.container(border=True):
        st.subheader("Profile Summary")
        st.caption("Based on your registered data")
        st.write("")

        h1, h2, h3, h4, h5 = st.columns(5)
        with h1: st.metric("Age", f"{age} yrs" if age != "N/A" else "N/A")
        with h2: st.metric("Height", f"{height} cm" if height != "N/A" else "N/A")
        with h3: st.metric("Weight", f"{weight} kg" if weight != "N/A" else "N/A")
        with h4: st.metric("BMI", f"{bmi} kg/m²" if bmi != "N/A" else "N/A")
        with h5: st.metric("Blood Group", blood_group)