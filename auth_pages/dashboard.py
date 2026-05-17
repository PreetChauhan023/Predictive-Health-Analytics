import streamlit as st
from streamlit_option_menu import option_menu
from auth_pages.home import page_home
from auth_pages.profile import page_profile
from auth_pages.chatbot import page_chatbot
from auth_pages.settings import page_settings
from auth_pages.dataset import page_dataset
from auth_pages.labreport import page_labreport
from auth_pages.about import page_about
from auth_pages.disease_outbreak import page_disease_outbreak_predictor
from auth_pages.risk_assessment import page_risk_assessment
from auth_pages.disease_prediction import page_disease_prediction
from auth_pages.consult import page_consult
from auth_pages.resource import page_resource_planning

LOGO_URL = "https://cdn-icons-png.flaticon.com/512/4807/4807695.png"

NAV_OPTIONS = [
    "Home",
    "Dataset Analysis",
    "Risk Assessment",
    "Disease Predictor",
    "AI Chatbot",
    "Lab Report Interpreter",
    "Disease Outbreak predictor",
    "Consult a Doctor",
    "Resource Planning",
    "Profile",
    "About",
    "Settings",
    "Logout"
]

NAV_ICONS = [
    "house-fill",
    "bar-chart-fill",
    "heart-pulse-fill",
    "clipboard2-heart-fill",
    "chat-dots-fill",
    "clipboard2-pulse-fill",
    "virus",
    "person-lines-fill",
    "gear-fill",
    "person-fill",
    "info-circle",
    "gear-fill",
    "box-arrow-right",
]

NAV_STYLES = {
    "container":{"background-color": "black"},
    "nav-link": {
                "font-size": "17px",
                "text-align": "left",
                "margin": "1px",
                "color": "white"
            },
            "nav-link-selected": {"background-color": "green"},
}

PAGE_ROUTES = {
    "Home":                        page_home,
    "Dataset Analysis":            page_dataset,
    "Risk Assessment":             page_risk_assessment,
    "Disease Predictor":           page_disease_prediction,
    "AI Chatbot":                  page_chatbot,
    "Lab Report Interpreter":      page_labreport,
    "Disease Outbreak predictor":  page_disease_outbreak_predictor,
    "Consult a Doctor":    page_consult,
    "Resource Planning": page_resource_planning,
    "Profile":                     page_profile,
    "About":                       page_about,
    "Settings":                    page_settings,
}

def _logout():
    st.query_params.clear()
    for key in ["user_email", "id_token", "username", "uid",
                "user_username", "user_phone", "user_dob", "user_gender",
                "gender", "q_category_index", "q_saved"]:
        st.session_state[key] = ""
    st.session_state.q_answers = {}
    st.session_state.page = "login"
    st.rerun()

def page_dashboard():
    with st.sidebar:
        col_logo, col_username = st.columns([1, 2])
        with col_logo:
            st.image(LOGO_URL, width=60)
        with col_username:
            username = st.session_state.get("username", "User")
            st.markdown(
                f"<span style='color:#2ECC71; font-size:19px; font-weight:bold; "
                f"font-style:italic; letter-spacing:2px;'>{username}</span>",
                unsafe_allow_html=True,
            )

        default_index = 0
        if st.session_state.get("_nav_to") in NAV_OPTIONS:
            default_index = NAV_OPTIONS.index(st.session_state["_nav_to"])
            st.session_state["_nav_to"] = None

        selected = option_menu(
            menu_title=None,
            options=NAV_OPTIONS,
            icons=NAV_ICONS,
            default_index=default_index,
            styles=NAV_STYLES,
        )


    if selected == "Logout":
        _logout()
        return


    if selected in PAGE_ROUTES:
        PAGE_ROUTES[selected]()