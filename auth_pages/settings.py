import re
import os
import pandas as pd
import streamlit as st
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGAL_DIR = os.path.join(BASE_DIR, "legal")
USER_DB   = "user_registration_data.csv"

# ── Helpers ───────────────────────────────────────────

def _read_legal(filename):
    path = os.path.join(LEGAL_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Content not available."


def _reauth(email, password):
    """Validates credentials — supports both hashed and legacy plain-text passwords."""
    import hashlib
    if os.path.exists(USER_DB):
        df = pd.read_csv(USER_DB, dtype={'email': str, 'password': str})
        row = df[df['email'] == email]
        if row.empty:
            return False
        stored = str(row.iloc[0]['password'])
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return stored == hashed or stored == password
    return False


# ── PDF Export ────────────────────────────────────────
def _generate_health_pdf(uid, email) -> bytes:
    """
    Build a complete health data PDF for the user and return it as bytes.
    Pulls from: user profile, risk history, and lab reports.
    """
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak,
    )

    W, H = A4
    BRAND_GREEN  = colors.HexColor("#1DB954")
    BRAND_DARK   = colors.HexColor("#0A2E1F")
    ACCENT_LIGHT = colors.HexColor("#E8F5EE")
    TEXT_GREY    = colors.HexColor("#4A5568")
    BORDER_GREY  = colors.HexColor("#CBD5E0")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=15*mm, bottomMargin=18*mm,
    )

    styles = getSampleStyleSheet()
    S = {
        "title": ParagraphStyle("title",
            fontSize=22, fontName="Helvetica-Bold",
            textColor=BRAND_DARK, spaceAfter=2),
        "subtitle": ParagraphStyle("subtitle",
            fontSize=10, fontName="Helvetica",
            textColor=TEXT_GREY, spaceAfter=6),
        "section": ParagraphStyle("section",
            fontSize=13, fontName="Helvetica-Bold",
            textColor=BRAND_DARK, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle("body",
            fontSize=10, fontName="Helvetica",
            textColor=TEXT_GREY, spaceAfter=3),
        "label": ParagraphStyle("label",
            fontSize=9, fontName="Helvetica-Bold",
            textColor=TEXT_GREY),
        "value": ParagraphStyle("value",
            fontSize=9, fontName="Helvetica",
            textColor=BRAND_DARK),
        "footer": ParagraphStyle("footer",
            fontSize=8, fontName="Helvetica",
            textColor=TEXT_GREY, alignment=1),
    }

    def section_header(title):
        return [
            Paragraph(title, S["section"]),
            HRFlowable(width="100%", thickness=1, color=BRAND_GREEN, spaceAfter=6),
        ]

    def kv_table(rows, col_widths=None):
        col_widths = col_widths or [65*mm, 95*mm]
        data = [[Paragraph(k, S["label"]), Paragraph(str(v), S["value"])] for k, v in rows]
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, ACCENT_LIGHT]),
            ("GRID",          (0, 0), (-1, -1), 0.4, BORDER_GREY),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    story = []

    # ── 1. Profile ────────────────────────────────────────────────────────────
    story += section_header("1. Personal Profile")
    profile_rows = []
    try:
        df_u = pd.read_csv(USER_DB, dtype=str)
        row  = df_u[df_u['email'] == email]
        if not row.empty:
            r = row.iloc[0]
            for col_label, field in [
                ("Full Name",     lambda r: f"{r.get('first_name','')} {r.get('last_name','')}".strip()),
                ("Email",         lambda r: r.get('email', '—')),
                ("Phone",         lambda r: r.get('phone', '—')),
                ("Date of Birth", lambda r: r.get('dob', '—')),
                ("Age",           lambda r: r.get('age', '—')),
                ("Gender",        lambda r: r.get('gender', r.get('Gender', '—'))),
                ("Blood Group",   lambda r: r.get('blood_group', '—')),
                ("Height (cm)",   lambda r: r.get('height_cm', '—')),
                ("Weight (kg)",   lambda r: r.get('weight_kg', '—')),
                ("Username",      lambda r: r.get('username', '—')),
                ("Member Since",  lambda r: r.get('created_at', '—')),
            ]:
                profile_rows.append((col_label, field(r)))
    except Exception:
        profile_rows = [("Error", "Could not load profile data.")]
    story.append(kv_table(profile_rows))

    # ── 2. Risk Assessment History ────────────────────────────────────────────
    story += section_header("2. Risk Assessment History")
    risk_history_path = os.path.join(BASE_DIR, "risk_history", f"{uid}.csv")
    try:
        if os.path.exists(risk_history_path):
            df_r = pd.read_csv(risk_history_path)
            df_r = df_r.sort_values("timestamp", ascending=False).head(10)
            headers = ["Date / Time", "Risk Score", "Risk Level", "Percentile"]
            data = [headers] + [
                [str(row_r.get("timestamp", "—")),
                 str(row_r.get("score", "—")),
                 str(row_r.get("risk_label", "—")),
                 f"{row_r.get('percentile', '—')}th"]
                for _, row_r in df_r.iterrows()
            ]
            t = Table(data, colWidths=[55*mm, 35*mm, 35*mm, 35*mm])
            t.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), BRAND_DARK),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS",(0, 1), (-1,-1), [colors.white, ACCENT_LIGHT]),
                ("GRID",          (0, 0), (-1, -1), 0.4, BORDER_GREY),
                ("LEFTPADDING",   (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
                ("TOPPADDING",    (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(t)
            if len(df_r) == 10:
                story.append(Paragraph("Showing 10 most recent assessments.", S["body"]))
        else:
            story.append(Paragraph("No risk assessments recorded yet.", S["body"]))
    except Exception as e:
        story.append(Paragraph(f"Could not load risk history: {e}", S["body"]))

    # ── 3. Lab Reports ────────────────────────────────────────────────────────
    story.append(PageBreak())
    story += section_header("3. Lab Reports")
    lab_csv = os.path.join(BASE_DIR, "data", "user_lab_reports.csv")
    lab_found = False

    LAB_DISPLAY = [
        ("analyzed_at",                          "Date Analyzed"),
        ("total_tests",                           "Total Tests"),
        ("normal",                                "Normal"),
        ("high",                                  "High"),
        ("low",                                   "Low"),
        ("borderline",                            "Borderline"),
        ("overall_interpretation",                "Overall Interpretation"),
        ("advice",                                "Advice"),
        ("cbc_hemoglobin_value",                  "Haemoglobin"),
        ("cbc_hemoglobin_status",                 "Haemoglobin Status"),
        ("cbc_wbc_count_value",                   "WBC Count"),
        ("cbc_wbc_count_status",                  "WBC Status"),
        ("cbc_platelet_count_value",              "Platelet Count"),
        ("cbc_platelet_count_status",             "Platelet Status"),
        ("lipid_profile_cholesterol_value",       "Total Cholesterol"),
        ("lipid_profile_cholesterol_status",      "Cholesterol Status"),
        ("lipid_profile_triglyceride_value",      "Triglycerides"),
        ("lipid_profile_triglyceride_status",     "Triglycerides Status"),
        ("lipid_profile_hdl_cholesterol_value",   "HDL Cholesterol"),
        ("lipid_profile_direct_ldl_value",        "LDL Cholesterol"),
        ("lipid_profile_direct_ldl_status",       "LDL Status"),
        ("blood_sugar_fasting_blood_sugar_value", "Fasting Blood Sugar"),
        ("blood_sugar_fasting_blood_sugar_status","Blood Sugar Status"),
        ("blood_sugar_hba1c_value",               "HbA1c"),
        ("blood_sugar_hba1c_status",              "HbA1c Status"),
        ("other_abo_type_value",                  "Blood Group (ABO)"),
        ("other_rh_(d)_type_value",               "Rh Factor"),
    ]

    try:
        if os.path.exists(lab_csv):
            df_lab = pd.read_csv(lab_csv, dtype=str)

            # Strip whitespace from uid/email columns and the lookup value
            uid_clean   = str(uid).strip().lower()
            email_clean = str(email).strip().lower()

            if 'uid' in df_lab.columns:
                df_lab['uid'] = df_lab['uid'].str.strip().str.lower()
                user_labs = df_lab[df_lab['uid'] == uid_clean]
            elif 'email' in df_lab.columns:
                df_lab['email'] = df_lab['email'].str.strip().str.lower()
                user_labs = df_lab[df_lab['email'] == email_clean]
            else:
                user_labs = pd.DataFrame()

            if not user_labs.empty:
                lab_found = True
                report_num = 0
                for _, lab_row in user_labs.iterrows():
                    # Skip rows that have no useful data at all
                    non_empty = [
                        v for k, v in lab_row.items()
                        if k not in ("uid", "email")
                        and str(v) not in ("nan", "", "Not Available")
                    ]
                    if not non_empty:
                        continue

                    report_num += 1
                    date_label = lab_row.get("analyzed_at", "")
                    header_text = f"Report {report_num}"
                    if date_label and str(date_label) not in ("nan", ""):
                        header_text += f"  —  {date_label}"
                    story.append(Paragraph(header_text, S["section"]))

                    lab_kv = []
                    for col, label in LAB_DISPLAY:
                        val = lab_row.get(col, "")
                        if val and str(val) not in ("nan", "", "Not Available"):
                            lab_kv.append((label, str(val)))

                    if lab_kv:
                        story.append(kv_table(lab_kv))
                    else:
                        story.append(Paragraph("No detailed test values recorded for this report.", S["body"]))
                    story.append(Spacer(1, 6*mm))

                if report_num == 0:
                    lab_found = False
        else:
            story.append(Paragraph(f"Lab reports file not found at: {lab_csv}", S["body"]))
    except Exception as e:
        story.append(Paragraph(f"Could not load lab reports: {e}", S["body"]))

    if not lab_found:
        story.append(Paragraph("No lab reports found for this account.", S["body"]))

    # ── 4. Disclaimer ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GREY))
    story.append(Spacer(1, 3*mm))
    story.append(Spacer(1, 2*mm))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _logout():
    st.query_params.clear()
    for key in ["user_email", "id_token", "username", "uid",
                "user_username", "user_phone", "user_dob", "user_gender",
                "gender", "q_category_index", "q_saved"]:
        st.session_state[key] = ""
    st.session_state.q_answers = {}
    st.session_state.page = "login"
    st.rerun()


def _is_valid_phone(phone):
    return bool(re.match(r'^(?:\+91|0)?[6-9]\d{9}$', phone))


# ── Main ──────────────────────────────────────────────
def page_settings():
    uid = st.session_state.get("uid", "")
    email = st.session_state.get("user_email", "")
    phone = st.session_state.get("user_phone", "N/A")

    st.title("Settings")
    st.write("Manage your account & preferences")
    st.divider()

    # ── 1. Profile Information ────────────────────────
    st.header("Personal Information")
    with st.container(border=True):

        # Phone row
        col_p_val, col_p_btn = st.columns([6, 1])
        with col_p_val:
            st.write("**Phone Number**")
            st.write(phone if phone else "Not set")
        with col_p_btn:
            if st.button("Edit", key="edit_phone_btn", type="primary"):
                st.session_state["_edit_phone"] = not st.session_state.get("_edit_phone", False)
                st.session_state["_edit_email"] = False
                st.rerun()

        if st.session_state.get("_edit_phone"):
            with st.form("form_phone", clear_on_submit=True):
                new_phone = st.text_input("New Phone Number", placeholder="+91XXXXXXXXXX")
                if st.form_submit_button("Save Phone", use_container_width=True):
                    if not new_phone:
                        st.error("Phone number is required.")
                    elif not _is_valid_phone(new_phone):
                        st.error("Enter a valid Indian phone number.")
                    elif new_phone == phone:
                        st.error("New number must differ from current.")
                    else:
                        try:
                            df = pd.read_csv(USER_DB, dtype={'phone': str, 'email': str})
                            if email in df['email'].values:
                                df.loc[df['email'] == email, 'phone'] = str(new_phone)
                                df.to_csv(USER_DB, index=False)

                                st.session_state.user_phone = str(new_phone)
                                st.session_state["_edit_phone"] = False
                                st.success("✅ Phone updated locally.")
                                st.rerun()
                            else:
                                st.error("User not found.")
                        except Exception as e:
                            st.error(f"Error: {e}")

        st.divider()

        # Email row
        col_e_val, col_e_btn = st.columns([6, 1])
        with col_e_val:
            st.write("**Email Address**")
            st.write(email if email else "Not set")
        with col_e_btn:
            if st.button("Edit", key="edit_email_btn", type="primary"):
                st.session_state["_edit_email"] = not st.session_state.get("_edit_email", False)
                st.session_state["_edit_phone"] = False
                st.rerun()

        if st.session_state.get("_edit_email"):
            st.caption("Enter your password to update your email address.")
            with st.form("form_email", clear_on_submit=False):
                current_pw = st.text_input("Current Password", type="password", placeholder="Confirm identity")
                new_email = st.text_input("New Email Address", placeholder="new@example.com")
                if st.form_submit_button("Update Email", use_container_width=True):
                    if not current_pw or not new_email:
                        st.error("All fields are required.")
                    elif new_email.lower() == email.lower():
                        st.error("New email must differ from current.")
                    else:
                        if _reauth(email, current_pw):
                            try:
                                # 1. Update Registration CSV
                                df_reg = pd.read_csv(USER_DB, dtype=str)
                                df_reg.loc[df_reg['email'] == email, 'email'] = new_email
                                df_reg.to_csv(USER_DB, index=False)

                                # 2. Update Health Responses CSV
                                health_path = os.path.join(BASE_DIR, HEALTH_DB)
                                if os.path.exists(health_path):
                                    df_health = pd.read_csv(health_path, dtype=str)
                                    if 'uid' in df_health.columns:
                                        df_health.loc[df_health['uid'] == email, 'uid'] = new_email
                                    if 'email' in df_health.columns:
                                        df_health.loc[df_health['email'] == email, 'email'] = new_email
                                    df_health.to_csv(health_path, index=False)

                                # 3. Update Chat History CSV
                                chat_path = os.path.join(BASE_DIR, "chat_history.csv")
                                if os.path.exists(chat_path):
                                    df_chat = pd.read_csv(chat_path, dtype=str)
                                    if 'uid' in df_chat.columns:
                                        df_chat.loc[df_chat['uid'] == email, 'uid'] = new_email
                                    if 'email' in df_chat.columns:
                                        df_chat.loc[df_chat['email'] == email, 'email'] = new_email
                                    df_chat.to_csv(chat_path, index=False)

                                # 4. Update Session State
                                st.session_state.user_email = new_email
                                st.session_state.uid = new_email
                                st.session_state["_edit_email"] = False
                                st.success("✅ Email updated in all records!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Sync Error: {e}")
                        else:
                            st.error("Password is incorrect.")

        st.divider()

        # Password row
        col_pw_val, col_pw_btn = st.columns([6, 1])
        with col_pw_val:
            st.write("**Password**")
            st.write("••••••••••••")
        with col_pw_btn:
            if st.button("Reset", key="reset_pw_btn", type="primary"):
                st.session_state["_reset_pw"] = not st.session_state.get("_reset_pw", False)
                st.rerun()

        if st.session_state.get("_reset_pw"):
            st.caption("Enter a new password below.")
            with st.form("form_reset_pw", clear_on_submit=True):
                new_pw = st.text_input("New Password", type="password")
                if st.form_submit_button("Update Password", use_container_width=True):
                    if not new_pw:
                        st.error("Password cannot be empty.")
                    else:
                        import hashlib
                        df = pd.read_csv(USER_DB, dtype=str)
                        df.loc[df['email'] == email, 'password'] = hashlib.sha256(new_pw.encode()).hexdigest()
                        df.to_csv(USER_DB, index=False)
                        st.success("Password updated!")
                        st.session_state["_reset_pw"] = False
                        st.rerun()

    # ── 2. Legal ──────────────────────────────────────
    st.header("Legal")
    for label, filename in [
        ("Privacy Policy", "privacy_policy.txt"),
        ("Terms of Service", "terms_of_service.txt"),
        ("Licences", "licences.txt"),
    ]:
        with st.expander(label):
            st.text(_read_legal(filename))

    # ── 3. Data Management (Export) ───────────────────
    st.header("Data Management")
    with st.container(border=True):
        st.write("Download your complete health data as a PDF ")
        if st.button("Generate PDF Report", key="prepare_export", use_container_width=True, type="primary"):
            with st.spinner("Building your report…"):
                try:
                    pdf_bytes = _generate_health_pdf(uid, email)
                    filename  = f"health_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.success("Report ready! Click the button above to download.")
                except Exception as e:
                    st.error(f"Failed to generate report: {e}")

    # ── 4. Account ────────────────────────────────────
    st.header("Account")
    with st.expander("Delete Account"):
        st.error("This action cannot be undone.")
        with st.form("form_delete", clear_on_submit=False):
            dpw = st.text_input("Password", type="password", placeholder="Confirm identity")
            dcon = st.text_input("Type your email address to confirm", placeholder=email)
            if st.form_submit_button("Permanently Delete My Account", use_container_width=True):
                if dcon.strip().lower() == email.strip().lower() and _reauth(email, dpw):
                    df_u = pd.read_csv(USER_DB, dtype=str)
                    df_u[df_u["email"] != email].to_csv(USER_DB, index=False)

                    csv_f = os.path.join(BASE_DIR, HEALTH_DB)
                    if os.path.exists(csv_f):
                        df_h = pd.read_csv(csv_f, dtype=str)
                        df_h[df_h["uid"] != uid].to_csv(csv_f, index=False)

                    st.success("Account deleted.")
                    _logout()
                else:
                    st.error("Incorrect details.")
