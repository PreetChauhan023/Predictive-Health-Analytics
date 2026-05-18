import streamlit as st


def page_about():
    st.title("Predictive Health Analytics")
    st.subheader("Risk Assessment & Disease Outbreak Forecasting")

    st.markdown("""
    This system leverages **advanced data analytics and machine learning** to assess
    individual health risks and forecast potential disease outbreaks.

    By analyzing **demographic data, medical history, lifestyle patterns, and environmental factors**,
    it generates **personalized risk assessments** and predicts **population-level health trends**.

    The goal is to enable **proactive, data-driven healthcare decisions** for individuals,
    healthcare professionals, and public health authorities.
    """)

    st.header("Problem vs Solution")
    col_problems, col_solutions = st.columns(2)

    with col_problems:
        st.markdown("### Existing Challenges")
        st.write("""
        * Healthcare systems rely on **reactive approaches** rather than prevention
        * Risk assessments are **manual and rule-based**
        * **Delayed reporting** affects outbreak response
        * Data sources are **fragmented and poorly integrated**
        * Lack of **personalized health predictions**
        """)

    with col_solutions:
        st.markdown("### Proposed Solution")
        st.write("""
        * **Machine Learning Models** for personalized risk analysis
        * **Early outbreak prediction** using population-level data
        * Integration of **multi-source health data**
        * **Risk classification** (Low / Medium / High)
        * **Proactive alerts & recommendations**
        """)

    # How it works — 4-step pipeline overview
    st.markdown("""
    <div style="background:#112D3B; border:1px solid #2F4F4F; border-radius:14px; padding:24px 28px;">
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:20px; text-align:center;">
            <div>
                <div style="font-size:28px; margin-bottom:8px;"></div>
                <div style="font-size:13px; font-weight:600; color:#E6F1F5; margin-bottom:6px;">1. Data Collection</div>
                <div style="font-size:12px; color:#9FB4BE; line-height:1.6;">
                    You enter your health parameters — vitals, lifestyle, medical history, and environmental factors.
                </div>
            </div>
            <div>
                <div style="font-size:28px; margin-bottom:8px;"></div>
                <div style="font-size:13px; font-weight:600; color:#E6F1F5; margin-bottom:6px;">2. ML Processing</div>
                <div style="font-size:12px; color:#9FB4BE; line-height:1.6;">
                    RandomForest models trained on 30,000 records process your inputs and generate predictions.
                </div>
            </div>
            <div>
                <div style="font-size:28px; margin-bottom:8px;"></div>
                <div style="font-size:13px; font-weight:600; color:#E6F1F5; margin-bottom:6px;">3. Risk Scoring</div>
                <div style="font-size:12px; color:#9FB4BE; line-height:1.6;">
                    A risk score (0–100) is assigned with a Low or Medium classification based on the dataset threshold.
                </div>
            </div>
            <div>
                <div style="font-size:28px; margin-bottom:8px;"></div>
                <div style="font-size:13px; font-weight:600; color:#E6F1F5; margin-bottom:6px;">4. Recommendations</div>
                <div style="font-size:12px; color:#9FB4BE; line-height:1.6;">
                    Personalised health recommendations and population context help you take preventive action.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.header("System Architecture")

    with st.expander("1. Health Data Collection"):
        st.write("""
        * Collects individual health parameters and demographic details
        * Aggregates environmental and population-level datasets
        """)

    with st.expander("2. Data Preprocessing"):
        st.write("""
        * Cleans and normalizes raw health data
        * Handles missing, inconsistent, and noisy data
        """)

    with st.expander("3. Individual Risk Assessment"):
        st.write("""
        * Uses machine learning models to evaluate personal health risks
        * Generates **risk scores and classifications**
        """)

    with st.expander("4. Disease Outbreak Forecasting"):
        st.write("""
        * Analyzes trends in population health data
        * Predicts potential outbreak patterns for early intervention
        """)

    with st.expander("5. Visualization Dashboard"):
        st.write("""
        * Displays health insights and outbreak trends
        * Provides intuitive dashboards for easy understanding
        """)

    with st.expander("6. Alerts & Recommendations"):
        st.write("""
        * Sends early warnings for high-risk individuals or regions
        * Suggests preventive healthcare measures
        """)

    st.header("Technologies Used")
    st.info("""
    **Frontend:** Streamlit, Plotly, Matplotlib, Seaborn
    **Backend:** Python, Pandas, NumPy, Scikit-Learn
    **Data Storage:** CSV / Excel
    """)

    st.write("Designed to support preventive healthcare and improve public health outcomes through intelligent analytics.")
    st.write("Project conceptualized, visualized, and developed by Preet Chauhan.")