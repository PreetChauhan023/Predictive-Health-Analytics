import os
import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import plotly.graph_objects as go

PLOTLY_BASE = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": dict(color="#0F172A"),
    "hovermode": "closest"
}

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "final_dataset.csv")


@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    df["Disease_Reported"]    = df["Disease_Reported"].fillna("Healthy")
    df["Alcohol_Consumption"] = df["Alcohol_Consumption"].fillna("None")
    df["Record_Date"]         = pd.to_datetime(df["Record_Date"], errors="coerce")
    df["Year"]                = df["Record_Date"].dt.year
    return df

def page_dataset():
    st.title("Dataset Analysis")
    st.divider()
    df = load_data()
    selected = option_menu(
        menu_title=None,
        options=["Dataset Explorer", "Overview", "Analytics"],
        icons=["table", "bar-chart", "graph-up"],
        orientation="horizontal",
        default_index=0,styles={
            "container": {
                "border": "1px solid green",
                "padding": "15px",
                "border-radius": "10px",
            },
                "nav-link": {
                "font-size": "18px",
                "text-align": "center",
                "margin": "0px",

            },
        "nav-link-selected":
            {"background-color": "green"},
            }
    )

    if selected == "Dataset Explorer":
        # ── Dataset KPIs ──────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{df.shape[0]:,}")
        c2.metric("Total Features", f"{df.shape[1]}")
        c3.metric("Missing Values", f"{df.isna().sum().sum()}")
        c4.metric("Date Range", "2020 – 2024")
        st.divider()
        # ── Advanced Filters ──────────────────────────
        st.subheader("Advanced Data Filtering")
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            disease_opts = ["All"] + sorted(df["Disease_Reported"].unique().tolist())
            sel_disease = st.selectbox("Filter by Disease", disease_opts, key="f_disease")
            risk_opts = ["All"] + sorted(df["Risk_Classification"].unique().tolist())
            sel_risk = st.selectbox("Filter by Risk Level", risk_opts, key="f_risk")
        with f_col2:
            state_opts = ["All"] + sorted(df["State"].unique().tolist())
            sel_state = st.selectbox("Filter by State", state_opts, key="f_state")
            gender_opts = ["All"] + sorted(df["Gender"].unique().tolist())
            sel_gender = st.selectbox("Filter by Gender", gender_opts, key="f_gender")
        with f_col3:
            region_opts = ["All"] + sorted(df["Region_Type"].unique().tolist())
            sel_region = st.selectbox("Filter by Region", region_opts, key="f_region")
            age_range = st.slider("Age Range",
                                  int(df["Age"].min()), int(df["Age"].max()),
                                  (int(df["Age"].min()), int(df["Age"].max())), key="f_age")

        if st.button("Reset Filters", use_container_width=True):
            for key in ["f_disease", "f_risk", "f_state", "f_gender", "f_region", "f_age"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # ── Apply Filters ─────────────────────────────
        filtered_df = df.copy()
        if sel_disease != "All":
            filtered_df = filtered_df[filtered_df["Disease_Reported"] == sel_disease]
        if sel_risk != "All":
            filtered_df = filtered_df[filtered_df["Risk_Classification"] == sel_risk]
        if sel_state != "All":
            filtered_df = filtered_df[filtered_df["State"] == sel_state]
        if sel_gender != "All":
            filtered_df = filtered_df[filtered_df["Gender"] == sel_gender]
        if sel_region != "All":
            filtered_df = filtered_df[filtered_df["Region_Type"] == sel_region]
        filtered_df = filtered_df[filtered_df["Age"].between(age_range[0], age_range[1])]

        st.divider()

        # ── Filtered KPIs ─────────────────────────────
        st.subheader("Filtered Results Summary")
        fk1, fk2, fk3, fk4, fk5 = st.columns(5)
        fk1.metric("Filtered Records", f"{len(filtered_df):,}",
                   f"{len(filtered_df) / len(df) * 100:.1f}% of total")
        fk2.metric("Avg Age",
                   f"{filtered_df['Age'].mean():.1f}" if len(filtered_df) else "—")
        fk3.metric("Avg BMI",
                   f"{filtered_df['BMI'].mean():.1f}" if len(filtered_df) else "—")
        fk4.metric("Avg Risk Score",
                   f"{filtered_df['Risk_Score'].mean():.1f}" if len(filtered_df) else "—")
        fk5.metric("High Risk",
                   f"{(filtered_df['Risk_Classification'] == 'High').sum():,}" if len(filtered_df) else "—")
        st.divider()

        # ── Filtered Dataset Table ────────────────────
        st.subheader("Filtered Dataset")
        st.dataframe(filtered_df, hide_index=True, use_container_width=True, height=400)
        st.divider()

        # ── Column Selector ───────────────────────────
        st.subheader("Filtered Results — Column View")
        all_columns = filtered_df.columns.tolist()
        column_selection = st.multiselect(
            "Select columns to view",
            options=["All"] + all_columns,
            default=["All"]
        )
        display_cols = all_columns if "All" in column_selection else column_selection
        if display_cols:
            st.dataframe(filtered_df[display_cols], use_container_width=True, height=600, hide_index=True)
        else:
            st.warning("Please select at least one column to display.")
        st.divider()

        # ── Download ──────────────────────────────────
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Filtered Data to CSV",
            data=csv,
            file_name="health_analytics_export.csv",
            mime="text/csv",
            use_container_width=True
        )

    elif selected == "Overview":
        st.divider()

        # KPIs
        total = len(df)
        high_risk = (df["Risk_Classification"] == "High").sum()
        avg_age = df["Age"].mean()
        avg_bmi = df["BMI"].mean()
        top_disease = df[df["Disease_Reported"] != "Healthy"]["Disease_Reported"].value_counts().idxmax()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Patients", f"{total:,}")

        k2.metric("Avg Age", f"{avg_age:.1f} yrs")
        k3.metric("Avg BMI", f"{avg_bmi:.1f}")
        k4.metric("Top Disease", top_disease)
        st.divider()

        # Disease summary table + risk pie
        st.subheader("Disease & Risk Breakdown")

        disease_summary = df.groupby("Disease_Reported").agg(
            Total_Patients=("Patient_ID", "count"),
            Avg_Risk_Score=("Risk_Score", "mean"),
            Avg_Age=("Age", "mean"),
            Avg_BMI=("BMI", "mean"),
        ).round(2)
        disease_summary["Risk Share %"] = (disease_summary["Total_Patients"] / total * 100).round(1)
        st.dataframe(
            disease_summary.style.background_gradient(subset=["Total_Patients"], cmap="YlOrRd"),
            use_container_width=True
        )

        risk_counts = df["Risk_Classification"].value_counts()
        fig_risk = px.pie(
            names=risk_counts.index, values=risk_counts.values,
            hole=0.4, title="Risk Classification Distribution",
            color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"]
        )
        fig_risk.update_layout(
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0F172A")
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        st.divider()

        # State analysis
        st.subheader("State & Region Analysis")
        s1, s2 = st.columns(2)
        with s1:
            state_counts = df.groupby("State")["Patient_ID"].count().reset_index()
            state_counts.columns = ["State", "Patients"]
            state_counts = state_counts.sort_values("Patients", ascending=False)
            fig_state = px.bar(
                state_counts, x="Patients", y="State", orientation="h",
                title="Patients per State", color="Patients",
                color_continuous_scale="Blues", text_auto=True
            )
            fig_state.update_layout(
                height=420,
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0F172A")
            )
            st.plotly_chart(fig_state, use_container_width=True)
        with s2:
            region_risk = df.groupby(["Region_Type", "Risk_Classification"]).size().reset_index(name="Count")
            fig_region = px.bar(
                region_risk, x="Region_Type", y="Count",
                color="Risk_Classification", title="Risk Distribution by Region",
                barmode="group",
                color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
            )
            fig_region.update_layout(
                height=420,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#0F172A")
            )
            st.plotly_chart(fig_region, use_container_width=True)
        st.divider()

        # Outbreak trend
        st.subheader("Disease Outbreak Trend (2020–2024)")
        trend = df.groupby(["Year", "Disease_Reported"]).size().reset_index(name="Cases")
        fig_trend = px.line(
            trend, x="Year", y="Cases", color="Disease_Reported",
            title="Yearly Disease Trend", markers=True
        )
        fig_trend.update_traces(line_width=2, marker_size=6)
        fig_trend.update_layout(
            height=450,
            legend=dict(font=dict(size=10, color="#0F172A")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0F172A")
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.divider()
        ch1, ch2 = st.columns(2)

        with ch1:
            # Risk score distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['Risk_Score'] <= 33]['Risk_Score'],
                nbinsx=20, name="Low Risk",
                marker_color="rgba(46,204,113,0.7)",
            ))
            fig.add_trace(go.Histogram(
                x=df[df['Risk_Score'] > 33]['Risk_Score'],
                nbinsx=15, name="Medium Risk",
                marker_color="rgba(243,156,18,0.7)",
            ))
            fig.add_vline(
                x=33, line_dash="dash", line_color="#E6F1F5", line_width=1,
                annotation_text=" Threshold (33)",
                annotation_font_color="#9FB4BE", annotation_font_size=11,
            )
            fig.update_layout(
                title="Risk Score Distribution — 30,000 Patients",
                barmode="overlay",
                xaxis_title="Risk Score",
                yaxis_title="Patients",
                height=320,
                showlegend=True,
                legend=dict(font=dict(color="#E6F1F5"), x=0.7, y=0.95),
                margin=dict(t=50, b=40, l=40, r=20),
                **PLOTLY_BASE,
            )
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            # Disease breakdown donut
            disease_counts = df[df['Disease_Reported'] != 'Healthy']['Disease_Reported'].value_counts()
            fig2 = go.Figure(go.Pie(
                labels=disease_counts.index.tolist(),
                values=disease_counts.values.tolist(),
                hole=0.52,
                textinfo="label+percent",
                textfont_size=11,
                marker=dict(
                    colors=["#2A9D8F","#E9C46A","#E76F51","#264653",
                            "#43AA8B","#F4A261","#577590","#90BE6D"],
                    line=dict(color="#0B1F2A", width=1.5),
                ),
                hovertemplate="%{label}: %{value:,} cases<extra></extra>",
            ))
            fig2.add_annotation(
                text=f"<b>{disease_counts.sum():,}</b><br><span style='font-size:10px'>cases</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="#E6F1F5"),
            )
            fig2.update_layout(
                title="Disease Case Distribution",
                height=320,
                showlegend=False,
                margin=dict(t=50, b=10, l=10, r=10),
                **PLOTLY_BASE,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Risk by state bar
        state_risk = df.groupby('State').agg(
            avg_score=('Risk_Score','mean'),
            medium_pct=('Risk_Score', lambda x: (x>33).mean()*100)
        ).round(1).reset_index().sort_values('avg_score', ascending=False)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=state_risk['State'],
            y=state_risk['avg_score'],
            marker=dict(
                color=state_risk['avg_score'],
                colorscale=[[0,'#2ECC71'],[0.5,'#E9C46A'],[1,'#E76F51']],
                showscale=False,
            ),
            text=[f"{v:.1f}" for v in state_risk['avg_score']],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg score: %{y:.1f}<extra></extra>",
        ))
        fig3.add_hline(
            y=33, line_dash="dash", line_color="#9FB4BE", line_width=1,
            annotation_text=" Risk threshold (33)",
            annotation_font_color="#9FB4BE", annotation_font_size=11,
        )
        fig3.update_layout(
            title="Average Risk Score by State",
            xaxis_tickangle=-25,
            yaxis=dict(title="Avg Risk Score", range=[0, 40]),
            height=340,
            margin=dict(t=50, b=80, l=40, r=20),
            **PLOTLY_BASE,
        )
        st.plotly_chart(fig3, use_container_width=True)


    elif selected == "Analytics":
        st.divider()

        # Sunburst
        st.subheader("Disease × Risk Hierarchical View")
        fig_sun = px.sunburst(
            df, path=["Risk_Classification", "Disease_Reported"],
            values="Risk_Score", color="Risk_Score",
            color_continuous_scale="RdYlGn_r",
            title="Risk Score by Disease and Classification"
        )
        # Added transparent background
        fig_sun.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sun, use_container_width=True)
        st.divider()

        # Lifestyle charts
        st.subheader("Lifestyle & Clinical Insights")
        l1, l2 = st.columns(2)
        with l1:
            smoking = df["Smoking_Status"].value_counts().reset_index()
            smoking.columns = ["Status", "Count"]
            fig_smoke = px.pie(smoking, names="Status", values="Count",
                               title="Smoking Status", hole=0.4)
            # Added transparent background
            fig_smoke.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_smoke, use_container_width=True)

        with l2:
            fig_bmi = px.box(
                df, x="Risk_Classification", y="BMI",
                color="Risk_Classification", title="BMI by Risk Level",
                color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
            )
            # Appended transparent background to existing layout call
            fig_bmi.update_layout(showlegend=False, height=400, paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bmi, use_container_width=True)
        st.divider()

        # Age distribution
        st.subheader("Age & Gender Distribution")
        ag1, ag2 = st.columns(2)
        with ag1:
            fig_age = px.histogram(df, x="Age", nbins=20,
                                   color="Gender", title="Age Distribution by Gender",
                                   barmode="overlay", opacity=0.7)
            # Added transparent background
            fig_age.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_age, use_container_width=True)

        with ag2:
            fig_scatter = px.scatter(df, x="Age", y="BMI", color="Risk_Classification",
                                     title="Age vs BMI by Risk Level", opacity=0.5,
                                     color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"})
            # Added transparent background
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)
        st.divider()

        # Disease bar
        st.subheader("Patient Count per Disease")
        disease_counts = df["Disease_Reported"].value_counts().reset_index()
        disease_counts.columns = ["Disease", "Count"]
        fig_bar = px.bar(
            disease_counts, x="Count", y="Disease", orientation="h",
            title="Patient Count per Disease", color="Count",
            color_continuous_scale="Blues", text_auto=True
        )
        # Appended transparent background to existing layout call
        fig_bar.update_layout(height=480, yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.divider()

        # Vaccination
        st.subheader("Vaccination Status")
        v1, v2 = st.columns(2)
        with v1:
            vacc = df["Vaccination_Status"].value_counts().reset_index()
            vacc.columns = ["Status", "Count"]
            fig_vacc = px.pie(vacc, names="Status", values="Count",
                              title="Vaccination Coverage", hole=0.4)
            # Added transparent background
            fig_vacc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_vacc, use_container_width=True)

        with v2:
            vacc_risk = df.groupby(["Vaccination_Status", "Risk_Classification"]).size().reset_index(name="Count")
            fig_vr = px.bar(vacc_risk, x="Vaccination_Status", y="Count",
                            color="Risk_Classification", title="Risk Level by Vaccination Status",
                            barmode="group",
                            color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"})
            # Added transparent background
            fig_vr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_vr, use_container_width=True)