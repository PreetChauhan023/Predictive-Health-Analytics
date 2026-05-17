import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(page_title="Health Intelligence System", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv('../final_dataset.csv')
    return df


df = load_data()

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=[
            "Home",
            "Dataset Analysis",
            "Disease Outbreak",
            "Resource Planning",
            "Vaccination Coverage",
            "Profile + Settings"
        ],
        icons=["house", "table", "activity", "hospital", "shield-check", "gear"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    st.title("Public Health Management System")
    st.write("Centralized monitoring and predictive analytics platform.")

elif selected == "Dataset Analysis":
    st.title("Data Exploration")


elif selected == "Disease Outbreak":
    st.title("disease outbreak")

elif selected == "Resource Planning":
    st.title("Resource Allocation")
    cols = st.columns(3)


elif selected == "Vaccination Coverage":
    st.title("Vaccination Coverage Analytics")

elif selected == "Profile + Settings":
    st.title("System Configuration")
