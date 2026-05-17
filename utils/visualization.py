import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def _status_icon(status):
    """Internal helper for table icons."""
    return {
        "Normal": "✅", "High": "🔴",
        "Low": "🔵", "Borderline": "⚠️"
    }.get(status, "❓")


def render_donut(counts):
    """Renders a Plotly donut chart based on test result counts."""
    labels = ["Normal", "High", "Low", "Borderline"]
    values = [
        counts.get("normal", 0),
        counts.get("high", 0),
        counts.get("low", 0),
        counts.get("borderline", 0)
    ]
    colors = ["#16A34A", "#DC2626", "#2563EB", "#F59E0B"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+value",
        textfont=dict(size=12),
        hovertemplate="%{label}: %{value} tests<extra></extra>"
    ))

    fig.update_layout(
        height=250,
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def render_results_table(categories):
    """Converts the JSON categories into a styled Streamlit Dataframe."""
    all_tests = []
    for cat, tests in categories.items():
        if not tests:
            continue
        for t in tests:
            all_tests.append({
                "Category": cat,
                "Test Name": t.get("name"),
                "Value": f"{t.get('value')} {t.get('unit', '')}",
                "Status": f"{_status_icon(t.get('status'))} {t.get('status')}",
                "Normal Range": f"{t.get('normal_min')} - {t.get('normal_max')} {t.get('unit', '')}",
                "Interpretation": t.get("meaning")
            })

    if not all_tests:
        st.warning("No biomarkers found to display.")
        return

    df = pd.DataFrame(all_tests)

    # Styling for the status column
    def color_status(val):
        if "High" in val: return 'color: #DC2626; font-weight: bold'
        if "Low" in val: return 'color: #2563EB; font-weight: bold'
        if "Borderline" in val: return 'color: #F59E0B; font-weight: bold'
        return 'color: #16A34A'

    st.dataframe(
        df.style.map(color_status, subset=['Status']),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn("System", width="small"),
            "Test Name": st.column_config.TextColumn("Biomarker", width="medium"),
            "Value": st.column_config.TextColumn("Your Result"),
            "Status": st.column_config.TextColumn("Status"),
            "Interpretation": st.column_config.TextColumn("Meaning", width="large")
        }
    )