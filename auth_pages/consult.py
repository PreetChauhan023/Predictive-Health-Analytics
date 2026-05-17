import streamlit as st
from datetime import datetime
import pandas as pd
def page_consult():
    st.title("Consult Medical Expert")
    st.write("Submit your case to our Primary Medical Expert for initial assessment")

    # Load Registration Data for Pre-filling
    try:
        reg_df = pd.read_csv('user_registration_data.csv')

        user_info = reg_df.iloc[-1]

        default_name = f"{user_info['first_name']} {user_info['last_name']}"
        default_email = user_info['email']
        default_phone = str(user_info['phone'])

    except Exception as e:
        st.error(f"CSV Error: {e}")
        default_name = st.session_state.get("username", "")
        default_email = st.session_state.get("user_email", "")
        default_phone = st.session_state.get("user_phone", "")

    # Appointment Booking Form
    st.subheader("Consultation Request Form")
    with st.form("appointment_form"):
        col1, col2 = st.columns(2)

        with col1:
            full_name = st.text_input("Patient Name", value=default_name)
            email = st.text_input("Email Address", value=default_email)
            phone = st.text_input("Phone Number", value=default_phone)

        with col2:
            app_date = st.date_input("Preferred Date", min_value=datetime.today())
            app_time = st.time_input("Preferred Time")
            #consult_mode = st.radio("Consultation Mode", ["Video Call", "In-Person Visit"], horizontal=True)

        file_upload = st.file_uploader("Upload Medical Reports (Optional)", type=['pdf', 'jpg', 'png'])

        submitted = st.form_submit_button("Submit Case for Review")

        if submitted:
            if full_name and email :
                st.success("Case Submitted Successfully.")
                st.info(f"""
                    **Next Steps:**
                    1. **Initial Review:** Our Medical Expert will review your data and the symptoms provided.
                    2. **Consultation:** You will be contacted at **{app_time}** on **{app_date}**.
                    3. **Pathfinding:** Following the consult, medical expert will decide further course of action if specialized treatment is necessary.
                    """)
            else:
                st.error("Please provide your name, email ")
