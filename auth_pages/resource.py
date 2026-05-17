"""
Resource Planning Simulator
────────────────────────────
Helps government bodies and hospitals estimate the medical resources
needed during a disease outbreak or high-risk health event.

Formulas are grounded in:
  • WHO Emergency Response Planning guidelines
  • India National Health Profile (NHP) capacity ratios
  • CDC outbreak modelling benchmarks
"""

import os
import json
from math import ceil
from datetime import date, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET     = os.path.join(BASE_DIR, "final_dataset.csv")

# ── Theme shorthand ───────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(color="#E6F1F5"),
)

# ── Disease profiles ──────────────────────────────────────────────────────────
# Each disease has epidemiological parameters derived from published outbreak data.
DISEASE_PROFILES = {
    "Influenza / Flu": {
        "attack_rate":        {"Mild": 0.08, "Moderate": 0.18, "Severe": 0.30, "Critical": 0.45},
        "hospitalization":    0.05,   # fraction of infected cases needing hospital
        "icu_rate":           0.008,  # fraction of infected needing ICU
        "ventilator_rate":    0.003,
        "avg_stay_days":      4,
        "mortality_rate":     0.001,
        "vaccine_needed":     True,
        "ppe_multiplier":     1.0,
        "icon": "",
    },
    "COVID-19 / Respiratory": {
        "attack_rate":        {"Mild": 0.06, "Moderate": 0.15, "Severe": 0.28, "Critical": 0.42},
        "hospitalization":    0.15,
        "icu_rate":           0.04,
        "ventilator_rate":    0.02,
        "avg_stay_days":      8,
        "mortality_rate":     0.012,
        "vaccine_needed":     True,
        "ppe_multiplier":     2.0,
        "icon": "",
    },
    "Dengue / Vector-Borne": {
        "attack_rate":        {"Mild": 0.04, "Moderate": 0.12, "Severe": 0.22, "Critical": 0.35},
        "hospitalization":    0.12,
        "icu_rate":           0.02,
        "ventilator_rate":    0.005,
        "avg_stay_days":      6,
        "mortality_rate":     0.005,
        "vaccine_needed":     False,
        "ppe_multiplier":     1.2,
        "icon": "",
    },
    "Cholera / Waterborne": {
        "attack_rate":        {"Mild": 0.10, "Moderate": 0.20, "Severe": 0.35, "Critical": 0.50},
        "hospitalization":    0.20,
        "icu_rate":           0.03,
        "ventilator_rate":    0.008,
        "avg_stay_days":      5,
        "mortality_rate":     0.015,
        "vaccine_needed":     True,
        "ppe_multiplier":     1.5,
        "icon": "",
    },
    "Tuberculosis (TB)": {
        "attack_rate":        {"Mild": 0.02, "Moderate": 0.06, "Severe": 0.12, "Critical": 0.20},
        "hospitalization":    0.30,
        "icu_rate":           0.05,
        "ventilator_rate":    0.015,
        "avg_stay_days":      30,
        "mortality_rate":     0.03,
        "vaccine_needed":     True,
        "ppe_multiplier":     2.5,
        "icon": "🫁",
    },
    "Malaria": {
        "attack_rate":        {"Mild": 0.05, "Moderate": 0.14, "Severe": 0.25, "Critical": 0.38},
        "hospitalization":    0.10,
        "icu_rate":           0.015,
        "ventilator_rate":    0.004,
        "avg_stay_days":      5,
        "mortality_rate":     0.006,
        "vaccine_needed":     False,
        "ppe_multiplier":     1.1,
        "icon": "🦟",
    },
    "General Outbreak (Unknown)": {
        "attack_rate":        {"Mild": 0.07, "Moderate": 0.16, "Severe": 0.28, "Critical": 0.42},
        "hospitalization":    0.12,
        "icu_rate":           0.025,
        "ventilator_rate":    0.008,
        "avg_stay_days":      6,
        "mortality_rate":     0.008,
        "vaccine_needed":     False,
        "ppe_multiplier":     1.5,
        "icon": "⚕",
    },
}

# ── India states list ─────────────────────────────────────────────────────────
INDIA_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "Jammu & Kashmir", "Ladakh", "Puducherry", "Chandigarh",
]

# ── WHO / NHP staffing ratios ──────────────────────────────────────────────────
# Per concurrent hospitalised patient
STAFF_RATIOS = {
    "Doctors per hospitalised patient":    1 / 5,    # 1 doctor per 5 ward patients
    "ICU doctors per ICU patient":         1 / 2,    # 1 doc per 2 ICU beds
    "Nurses per hospitalised patient":     1 / 3,
    "ICU nurses per ICU patient":          1 / 1,    # 1:1 in ICU
    "Paramedics per hospitalised patient": 1 / 8,
    "Lab technicians":                     1 / 20,
    "Ambulances per 1000 cases":           1 / 1000,
}

# ── Color palette ─────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "Mild":     "#2ECC71",
    "Moderate": "#F39C12",
    "Severe":   "#E67E22",
    "Critical": "#E74C3C",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE CALCULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_resources(
    population:    int,
    severity:      str,
    disease:       str,
    horizon_days:  int,
    high_risk_pct: float,
    buffer_pct:    float = 20.0,
) -> dict:
    """
    Returns a dict with all resource estimates and intermediate values.

    Parameters
    ----------
    population    : total at-risk population
    severity      : Mild / Moderate / Severe / Critical
    disease       : key in DISEASE_PROFILES
    horizon_days  : planning period in days
    high_risk_pct : percentage of population classified as High-risk (0–100)
    buffer_pct    : surge / safety buffer percentage
    """
    p   = DISEASE_PROFILES[disease]
    buf = 1 + buffer_pct / 100

    # ── Effective population (high-risk cohort has 2× attack susceptibility) ──
    low_risk_pop  = population * ((100 - high_risk_pct) / 100)
    high_risk_pop = population * (high_risk_pct / 100)
    attack        = p["attack_rate"][severity]
    infected      = low_risk_pop * attack + high_risk_pop * attack * 2
    infected      = int(infected)

    # ── Case flow ──────────────────────────────────────────────────────────────
    # Spread infections across the horizon using a simple epidemic curve peak
    # at ~30 % of the period (rising limb) → peak → decline.
    peak_fraction   = 0.20          # fraction of total cases active at peak
    peak_active     = int(infected * peak_fraction)
    hospitalised    = int(peak_active * p["hospitalization"])
    icu             = int(peak_active * p["icu_rate"])
    ventilated      = int(peak_active * p["ventilator_rate"])
    deaths_total    = int(infected   * p["mortality_rate"])
    recovered_total = infected - deaths_total

    avg_stay        = p["avg_stay_days"]
    # Total bed-days needed over horizon
    total_bed_days  = int(infected * p["hospitalization"] * avg_stay)

    # ── Infrastructure ─────────────────────────────────────────────────────────
    beds_general    = ceil(hospitalised * buf)
    beds_icu        = ceil(icu          * buf)
    beds_isolation  = ceil(hospitalised * 0.30 * buf)   # 30 % in isolation
    ventilators     = ceil(ventilated   * buf)
    oxygen_units    = ceil((hospitalised + icu) * 0.60 * buf)  # 60 % on O₂

    # ── Staffing ───────────────────────────────────────────────────────────────
    doctors_ward  = ceil(hospitalised * STAFF_RATIOS["Doctors per hospitalised patient"]  * buf)
    doctors_icu   = ceil(icu          * STAFF_RATIOS["ICU doctors per ICU patient"]       * buf)
    doctors_total = doctors_ward + doctors_icu

    nurses_ward   = ceil(hospitalised * STAFF_RATIOS["Nurses per hospitalised patient"]   * buf)
    nurses_icu    = ceil(icu          * STAFF_RATIOS["ICU nurses per ICU patient"]        * buf)
    nurses_total  = nurses_ward + nurses_icu

    paramedics    = ceil(hospitalised * STAFF_RATIOS["Paramedics per hospitalised patient"] * buf)
    lab_techs     = ceil(infected     * STAFF_RATIOS["Lab technicians"]                     * buf)
    ambulances    = ceil(infected / 1000 * buf)

    # ── Medicines & consumables (units over the full horizon) ──────────────────
    ppe_sets        = ceil(infected * p["ppe_multiplier"] * buf)
    test_kits       = ceil(infected * 3 * buf)            # ~3 tests per case
    iv_fluids_L     = ceil(hospitalised * avg_stay * 2 * buf)  # 2 L/day/patient
    analgesics      = ceil(hospitalised * avg_stay * buf)
    antibiotics     = ceil(hospitalised * 0.40 * avg_stay * buf)
    vaccines_doses  = ceil(population * 0.70 * buf) if p["vaccine_needed"] else 0

    # ── Budget estimate (INR, rough order-of-magnitude) ───────────────────────
    # Based on published ICMR / MoH per-patient cost estimates
    cost_per_ward_day     = 3_000    # ₹
    cost_per_icu_day      = 15_000   # ₹
    cost_bed_infra        = beds_general * 50_000 + beds_icu * 2_00_000
    cost_staff_monthly    = (doctors_total * 1_00_000 + nurses_total * 40_000 + paramedics * 25_000)
    cost_medicines        = ppe_sets * 500 + test_kits * 200 + iv_fluids_L * 30 + analgesics * 10 + antibiotics * 80
    cost_ops              = int(infected * p["hospitalization"] * avg_stay * cost_per_ward_day
                               + icu * avg_stay * cost_per_icu_day)
    cost_vaccines         = vaccines_doses * 300 if p["vaccine_needed"] else 0
    total_budget_estimate = cost_bed_infra + cost_staff_monthly + cost_medicines + cost_ops + cost_vaccines

    return {
        # Inputs echoed
        "disease":      disease,
        "severity":     severity,
        "population":   population,
        "horizon_days": horizon_days,
        "buffer_pct":   buffer_pct,
        # Case flow
        "infected":           infected,
        "peak_active":        peak_active,
        "hospitalised":       hospitalised,
        "icu":                icu,
        "ventilated":         ventilated,
        "deaths_total":       deaths_total,
        "recovered_total":    recovered_total,
        "total_bed_days":     total_bed_days,
        # Infrastructure
        "beds_general":       beds_general,
        "beds_icu":           beds_icu,
        "beds_isolation":     beds_isolation,
        "ventilators":        ventilators,
        "oxygen_units":       oxygen_units,
        # Staffing
        "doctors_total":      doctors_total,
        "nurses_total":       nurses_total,
        "paramedics":         paramedics,
        "lab_techs":          lab_techs,
        "ambulances":         ambulances,
        # Medicines
        "ppe_sets":           ppe_sets,
        "test_kits":          test_kits,
        "iv_fluids_L":        iv_fluids_L,
        "analgesics":         analgesics,
        "antibiotics":        antibiotics,
        "vaccines_doses":     vaccines_doses,
        # Budget
        "total_budget_INR":   total_budget_estimate,
    }


def simulate_all_scenarios(population, disease, horizon_days, high_risk_pct, buffer_pct):
    """Run the calculator for all 4 severity levels for comparison."""
    return {s: calculate_resources(population, s, disease, horizon_days, high_risk_pct, buffer_pct)
            for s in ["Mild", "Moderate", "Severe", "Critical"]}


# ═══════════════════════════════════════════════════════════════════════════════
#  EPIDEMIC CURVE  (simple SIR-ish curve for visualisation only)
# ═══════════════════════════════════════════════════════════════════════════════

def epidemic_curve(infected_total: int, horizon_days: int, severity: str):
    """
    Returns (days_array, daily_new_cases, daily_hospitalisations).
    Uses a lognormal shape — realistic rising/falling curve.
    """
    days   = np.arange(1, horizon_days + 1)
    peak   = horizon_days * 0.30            # peak at 30 % of horizon
    sigma  = 0.40

    raw    = np.exp(-(np.log(days / peak) ** 2) / (2 * sigma ** 2)) / (days * sigma * np.sqrt(2 * np.pi))
    raw    = raw / raw.sum()                # normalise to sum = 1
    daily  = (raw * infected_total).astype(int)
    daily[-1] += infected_total - daily.sum()   # fix rounding

    hosp_rate = DISEASE_PROFILES[severity]["hospitalization"] if severity in DISEASE_PROFILES \
                else 0.12
    # Use severity key from disease profile
    return days, daily, (daily * hosp_rate).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _scenario_comparison_chart(scenarios: dict, metric: str, label: str):
    sevs   = list(scenarios.keys())
    values = [scenarios[s][metric] for s in sevs]
    colors = [SEVERITY_COLORS[s] for s in sevs]

    fig = go.Figure(go.Bar(
        x=sevs, y=values, marker_color=colors,
        text=[f"{v:,}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=label,
        yaxis=dict(title=label, range=[0, max(values) * 1.25]),
        height=300,
        margin=dict(t=40, b=10, l=20, r=20),
        **PLOTLY_BASE,
    )
    return fig


def _epidemic_curve_chart(days, daily, hosp, disease, severity):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=daily, mode="lines", name="New Cases/day",
        fill="tozeroy", line=dict(color="#3498DB", width=2),
        fillcolor="rgba(52,152,219,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=hosp, mode="lines", name="Hospitalisations/day",
        fill="tozeroy", line=dict(color="#E74C3C", width=2),
        fillcolor="rgba(231,76,60,0.12)",
    ))
    fig.update_layout(
        title=f"Projected Epidemic Curve — {disease} ({severity})",
        xaxis=dict(title="Day"),
        yaxis=dict(title="Daily Count"),
        height=350,
        legend=dict(font=dict(color="#E6F1F5")),
        **PLOTLY_BASE,
    )
    return fig


def _resource_breakdown_radar(r: dict):
    # Normalise each resource to a 0-100 scale for radar
    max_vals = {
        "Beds (General)":    5000,
        "ICU Beds":          500,
        "Ventilators":       200,
        "Doctors":           1000,
        "Nurses":            2000,
        "Ambulances":        50,
        "PPE Sets (000s)":   100,
        "Test Kits (000s)":  200,
    }
    actual = {
        "Beds (General)":   r["beds_general"],
        "ICU Beds":         r["beds_icu"],
        "Ventilators":      r["ventilators"],
        "Doctors":          r["doctors_total"],
        "Nurses":           r["nurses_total"],
        "Ambulances":       r["ambulances"],
        "PPE Sets (000s)":  r["ppe_sets"] / 1000,
        "Test Kits (000s)": r["test_kits"] / 1000,
    }

    cats  = list(max_vals.keys())
    norms = [min(actual[c] / max_vals[c] * 100, 100) for c in cats]
    cats_c = cats + [cats[0]]
    vals_c = norms + [norms[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=cats_c, fill="toself",
        line=dict(color="#2A9D8F", width=2),
        fillcolor="rgba(42,157,143,0.20)",
        name="Resource Load",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#9FB4BE"), gridcolor="#2F4F4F"),
            angularaxis=dict(tickfont=dict(color="#E6F1F5"), gridcolor="#2F4F4F"),
        ),
        title="Resource Demand Profile",
        height=380,
        **PLOTLY_BASE,
    )
    return fig


def _staffing_donut(r: dict):
    labels = ["Doctors", "Nurses", "Paramedics", "Lab Technicians"]
    values = [r["doctors_total"], r["nurses_total"], r["paramedics"], r["lab_techs"]]
    colors = ["#3498DB", "#2ECC71", "#E67E22", "#9B59B6"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.50,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Staffing Mix",
        height=300,
        showlegend=False,
        margin=dict(t=40, b=0, l=0, r=0),
        **PLOTLY_BASE,
    )
    return fig


def _timeline_bar(r: dict, horizon_days: int):
    """Shows when resources need to be mobilised — phased timeline."""
    phases = {
        "Immediate (Week 1)":    0.25,
        "Short-term (Month 1)":  0.55,
        "Medium-term (3 months)": 0.20,
    }
    resources = ["beds_general", "beds_icu", "doctors_total", "nurses_total",
                 "ppe_sets", "test_kits"]
    labels    = ["Gen. Beds", "ICU Beds", "Doctors", "Nurses", "PPE (00s)", "Test Kits (00s)"]
    scale     = [1, 1, 1, 1, 100, 100]

    colors    = ["#3498DB", "#E74C3C", "#2ECC71"]
    fig       = go.Figure()

    for (phase, frac), color in zip(phases.items(), colors):
        fig.add_trace(go.Bar(
            name=phase,
            x=labels,
            y=[ceil(r[k] * frac / s) for k, s in zip(resources, scale)],
            marker_color=color,
        ))

    fig.update_layout(
        title="Phased Mobilisation Timeline",
        barmode="stack",
        yaxis=dict(title="Units required"),
        height=350,
        legend=dict(font=dict(color="#E6F1F5")),
        **PLOTLY_BASE,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _result_to_df(r: dict) -> pd.DataFrame:
    rows = [
        ("Disease",              r["disease"],                  ""),
        ("Severity Scenario",    r["severity"],                  ""),
        ("Population",           f"{r['population']:,}",         "People"),
        ("Planning Horizon",     r["horizon_days"],              "Days"),
        ("", "", ""),
        ("── CASE PROJECTIONS ──", "", ""),
        ("Estimated Infected",   f"{r['infected']:,}",           "Cases"),
        ("Peak Active Cases",    f"{r['peak_active']:,}",        "Concurrent"),
        ("Hospitalised (Peak)",  f"{r['hospitalised']:,}",       "Patients"),
        ("ICU Cases (Peak)",     f"{r['icu']:,}",                "Patients"),
        ("Ventilated (Peak)",    f"{r['ventilated']:,}",         "Patients"),
        ("Estimated Deaths",     f"{r['deaths_total']:,}",       "Total"),
        ("", "", ""),
        ("── INFRASTRUCTURE ──", "", ""),
        ("General Hospital Beds",f"{r['beds_general']:,}",       "Beds"),
        ("ICU Beds",             f"{r['beds_icu']:,}",           "Beds"),
        ("Isolation Wards",      f"{r['beds_isolation']:,}",     "Beds"),
        ("Ventilators",          f"{r['ventilators']:,}",        "Units"),
        ("Oxygen Support Units", f"{r['oxygen_units']:,}",       "Units"),
        ("Total Bed-Days",       f"{r['total_bed_days']:,}",     "Bed-Days"),
        ("", "", ""),
        ("── STAFFING ──", "", ""),
        ("Doctors Required",     f"{r['doctors_total']:,}",      "Personnel"),
        ("Nurses Required",      f"{r['nurses_total']:,}",       "Personnel"),
        ("Paramedics",           f"{r['paramedics']:,}",         "Personnel"),
        ("Lab Technicians",      f"{r['lab_techs']:,}",          "Personnel"),
        ("Ambulances",           f"{r['ambulances']:,}",         "Vehicles"),
        ("", "", ""),
        ("── MEDICINES & CONSUMABLES ──", "", ""),
        ("PPE Sets",             f"{r['ppe_sets']:,}",           "Sets"),
        ("Diagnostic Test Kits", f"{r['test_kits']:,}",          "Kits"),
        ("IV Fluids",            f"{r['iv_fluids_L']:,}",        "Litres"),
        ("Analgesics",           f"{r['analgesics']:,}",         "Units"),
        ("Antibiotics",          f"{r['antibiotics']:,}",        "Units"),
        ("Vaccine Doses",        f"{r['vaccines_doses']:,}" if r['vaccines_doses'] else "N/A", "Doses"),
        ("", "", ""),
        ("── BUDGET ──", "", ""),
        ("Total Estimated Cost",  f"₹ {r['total_budget_INR']:,.0f}", "INR (Approximate)"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value", "Unit"])


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

def page_resource_planning():
    st.title("Resource Planning Simulator")
    st.caption(
        "Simulate medical resource needs for disease outbreaks. "
        "Designed for government bodies, district health officers, and hospital administrators."
    )
    st.divider()

    # ── Load dataset for population risk context ───────────────────────────────
    @st.cache_data
    def _load_ds():
        if os.path.exists(DATASET):
            return pd.read_csv(DATASET)
        return pd.DataFrame()

    df_pop = _load_ds()

    # ─────────────────────────────────────────────────────────────────────────
    #  TABS
    # ─────────────────────────────────────────────────────────────────────────
    tab_input, tab_forecast, tab_staffing, tab_scenarios, tab_export = st.tabs([
        "Configure", "Resource Forecast", "Staffing Plan",
        "Scenario Comparison", "Export Report"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — CONFIGURE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_input:
        st.subheader("Outbreak Parameters")
        st.write("Configure the simulation inputs. All fields are used to calculate resource requirements.")
        st.write("")

        c1, c2, c3 = st.columns(3)

        with c1:
            with st.container(border=True):
                st.markdown("**Geography**")
                state = st.selectbox("State / Region", INDIA_STATES, index=INDIA_STATES.index("Gujarat"))
                region_type = st.selectbox("Region Type", ["Urban", "Semi-Urban", "Rural"])
                population = st.number_input(
                    "At-Risk Population",
                    min_value=1_000, max_value=50_000_000,
                    value=500_000, step=10_000,
                    help="Total population in the affected area"
                )

        with c2:
            with st.container(border=True):
                st.markdown("**Disease & Severity**")
                disease = st.selectbox(
                    "Disease / Outbreak Type",
                    list(DISEASE_PROFILES.keys()),
                )
                severity = st.select_slider(
                    "Outbreak Severity",
                    options=["Mild", "Moderate", "Severe", "Critical"],
                    value="Moderate",
                )
                profile = DISEASE_PROFILES[disease]
                attack_rate = profile["attack_rate"][severity]
                st.info(
                    f"{profile['icon']} **Attack Rate:** {attack_rate*100:.0f}% | "
                    f"**Hosp. Rate:** {profile['hospitalization']*100:.0f}% | "
                    f"**ICU Rate:** {profile['icu_rate']*100:.1f}%"
                )

        with c3:
            with st.container(border=True):
                st.markdown("**Planning Parameters**")
                horizon_option = st.radio(
                    "Planning Horizon",
                    ["1 Week (7 days)", "1 Month (30 days)", "3 Months (90 days)", "6 Months (180 days)"],
                    index=1,
                )
                horizon_map   = {"1 Week (7 days)": 7, "1 Month (30 days)": 30,
                                  "3 Months (90 days)": 90, "6 Months (180 days)": 180}
                horizon_days  = horizon_map[horizon_option]

                buffer_pct = st.slider(
                    "Surge / Safety Buffer (%)",
                    min_value=0, max_value=50, value=20, step=5,
                    help="Extra capacity over minimum estimate — WHO recommends 20–30%"
                )

        st.write("")
        st.subheader("Risk Distribution")
        st.write("Set the estimated percentage of the population in each risk category.")
        st.caption("These are pre-filled from the dataset if available. Adjust manually as needed.")

        # Pre-fill from dataset if possible
        default_high = 12.0
        if not df_pop.empty and "Risk_Classification" in df_pop.columns:
            vc = df_pop["Risk_Classification"].value_counts(normalize=True) * 100
            default_high = round(float(vc.get("High", 12.0)), 1)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            high_risk_pct = st.slider("High Risk (%)",  0.0, 100.0, default_high, 0.5)
        with rc2:
            med_risk_pct  = st.slider("Medium Risk (%)", 0.0, 100.0,
                                       min(50.0, 100.0 - high_risk_pct), 0.5)
        with rc3:
            low_risk_pct  = max(0.0, 100.0 - high_risk_pct - med_risk_pct)
            st.metric("Low Risk (%)", f"{low_risk_pct:.1f}%")

        st.write("")


        run_sim = st.button(
                "▶ Run Simulation", type="primary", use_container_width=True
            )

        if run_sim:
            with st.spinner("Running simulation..."):
                result    = calculate_resources(
                    population, severity, disease, horizon_days, high_risk_pct, buffer_pct
                )
                scenarios = simulate_all_scenarios(
                    population, disease, horizon_days, high_risk_pct, buffer_pct
                )
                st.session_state["_sim_result"]    = result
                st.session_state["_sim_scenarios"] = scenarios
                st.session_state["_sim_params"]    = {
                    "state": state, "region_type": region_type,
                    "disease": disease, "severity": severity,
                    "population": population, "horizon_days": horizon_days,
                    "buffer_pct": buffer_pct, "high_risk_pct": high_risk_pct,
                }
            st.success("Simulation complete! View results in the tabs above.")

    # ── Guard: require simulation to have been run ─────────────────────────────
    result    = st.session_state.get("_sim_result")
    scenarios = st.session_state.get("_sim_scenarios")
    params    = st.session_state.get("_sim_params", {})

    def _not_ready():
        st.info("Configure and run the simulation in the **⚙️ Configure** tab first.")

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — RESOURCE FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    with tab_forecast:
        if not result:
            _not_ready()
        else:
            sev_color = SEVERITY_COLORS[result["severity"]]

            # ── Headline banner ───────────────────────────────────────────────
            st.container(border=True).markdown(
                f"<div style='text-align:center;padding:6px'>"
                f"<span style='font-size:17px;color:{sev_color};font-weight:700'>"
                f"{DISEASE_PROFILES[result['disease']]['icon']} &nbsp;"
                f"{result['disease']} &nbsp;·&nbsp; "
                f"{result['severity']} Outbreak &nbsp;·&nbsp; "
                f"Population: {result['population']:,} &nbsp;·&nbsp; "
                f"Horizon: {result['horizon_days']} days"
                f"</span></div>",
                unsafe_allow_html=True,
            )
            st.write("")

            # ── KPI Row 1 — Cases ─────────────────────────────────────────────
            st.subheader("Projected Cases")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Infected",    f"{result['infected']:,}")
            k2.metric("Peak Active",       f"{result['peak_active']:,}")
            k3.metric("Hospitalised",      f"{result['hospitalised']:,}", "at peak")
            k4.metric("ICU Cases",         f"{result['icu']:,}",          "at peak")
            k5.metric("Estimated Deaths",  f"{result['deaths_total']:,}")
            st.divider()

            # ── Epidemic Curve ────────────────────────────────────────────────
            days, daily, hosp = epidemic_curve(
                result["infected"], result["horizon_days"], result["severity"]
            )
            st.plotly_chart(
                _epidemic_curve_chart(days, daily, hosp, result["disease"], result["severity"]),
                use_container_width=True,
            )
            st.divider()

            # ── KPI Row 2 — Infrastructure ────────────────────────────────────
            st.subheader("Infrastructure Requirements")
            i1, i2, i3, i4, i5, i6 = st.columns(6)
            i1.metric("General Beds",      f"{result['beds_general']:,}")
            i2.metric("ICU Beds",          f"{result['beds_icu']:,}")
            i3.metric("Isolation Wards",   f"{result['beds_isolation']:,}")
            i4.metric("Ventilators",       f"{result['ventilators']:,}")
            i5.metric("Oxygen Units",      f"{result['oxygen_units']:,}")
            i6.metric("Total Bed-Days",    f"{result['total_bed_days']:,}")
            st.divider()

            # ── Medicine KPIs ─────────────────────────────────────────────────
            st.subheader("Medicines & Consumables")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("PPE Sets",          f"{result['ppe_sets']:,}")
            m2.metric("Test Kits",         f"{result['test_kits']:,}")
            m3.metric("IV Fluids (L)",     f"{result['iv_fluids_L']:,}")
            m4.metric("Analgesics",        f"{result['analgesics']:,}")
            m5.metric("Antibiotics",       f"{result['antibiotics']:,}")
            m6.metric("Vaccine Doses",
                      f"{result['vaccines_doses']:,}" if result["vaccines_doses"] else "N/A")
            st.divider()

            # ── Radar + Timeline ─────────────────────────────────────────────
            r1, r2 = st.columns(2)
            with r1:
                st.plotly_chart(_resource_breakdown_radar(result), use_container_width=True)
            with r2:
                st.plotly_chart(_timeline_bar(result, result["horizon_days"]), use_container_width=True)
            st.divider()

            # ── Budget ────────────────────────────────────────────────────────
            with st.container(border=True):
                st.subheader("💰 Estimated Budget Requirement")
                st.caption("Rough order-of-magnitude estimate based on ICMR / MoH published cost benchmarks.")
                st.metric(
                    "Total Budget (INR)",
                    f"₹ {result['total_budget_INR']:,.0f}",
                    f"≈ ₹ {result['total_budget_INR']/1e7:.1f} Crore"
                )

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — STAFFING PLAN
    # ══════════════════════════════════════════════════════════════════════════
    with tab_staffing:
        if not result:
            _not_ready()
        else:
            st.subheader("Human Resource Requirements")
            st.caption(
                "Based on WHO staffing ratios with surge buffer. "
                "ICU doctors and nurses are counted separately from ward staff."
            )
            st.write("")

            # KPIs
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Doctors",         f"{result['doctors_total']:,}")
            s2.metric("Nurses",          f"{result['nurses_total']:,}")
            s3.metric("Paramedics",      f"{result['paramedics']:,}")
            s4.metric("Lab Technicians", f"{result['lab_techs']:,}")
            s5.metric("Ambulances",      f"{result['ambulances']:,}")
            st.divider()

            # Donut + detailed table
            sd1, sd2 = st.columns([1, 2])
            with sd1:
                st.plotly_chart(_staffing_donut(result), use_container_width=True)

            with sd2:
                staff_df = pd.DataFrame([
                    {"Role": "Senior Doctors (Ward)",   "Count": ceil(result["doctors_total"] * 0.30), "Deployment": "General Ward"},
                    {"Role": "Junior Doctors (Ward)",   "Count": ceil(result["doctors_total"] * 0.40), "Deployment": "General Ward"},
                    {"Role": "ICU Intensivists",        "Count": ceil(result["doctors_total"] * 0.30), "Deployment": "ICU"},
                    {"Role": "Head Nurses",             "Count": ceil(result["nurses_total"]  * 0.15), "Deployment": "Ward / ICU"},
                    {"Role": "Staff Nurses (Ward)",     "Count": ceil(result["nurses_total"]  * 0.55), "Deployment": "General Ward"},
                    {"Role": "ICU Nurses",              "Count": ceil(result["nurses_total"]  * 0.30), "Deployment": "ICU"},
                    {"Role": "Paramedics / EMTs",       "Count": result["paramedics"],                  "Deployment": "ER / Ambulance"},
                    {"Role": "Lab Technicians",         "Count": result["lab_techs"],                   "Deployment": "Pathology / Radiology"},
                    {"Role": "Ambulance Drivers",       "Count": result["ambulances"],                  "Deployment": "Field"},
                ])
                st.dataframe(staff_df, hide_index=True, use_container_width=True, height=320)

            st.divider()

            # Shift planning estimate
            with st.container(border=True):
                st.subheader("Shift Planning Estimate")
                st.caption("3-shift model (8 hrs each). Total personnel = displayed × 3 to cover all shifts.")
                sp1, sp2, sp3 = st.columns(3)
                sp1.metric("Doctors per shift",    f"{ceil(result['doctors_total'] / 3):,}")
                sp2.metric("Nurses per shift",     f"{ceil(result['nurses_total']  / 3):,}")
                sp3.metric("Paramedics per shift", f"{ceil(result['paramedics']    / 3):,}")

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — SCENARIO COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    with tab_scenarios:
        if not scenarios:
            _not_ready()
        else:
            st.subheader("Scenario Comparison All Severity Levels")
            st.write(
                "Compare resource requirements across Mild → Critical scenarios "
                f"for **{params.get('disease','the selected disease')}**, "
                f"population **{params.get('population', 0):,}**, "
                f"{params.get('horizon_days', 30)}-day horizon."
            )
            st.divider()

            # Summary comparison table
            rows = []
            for sev, r in scenarios.items():
                rows.append({
                    "Scenario":        sev,
                    "Infected":        f"{r['infected']:,}",
                    "Hospitalised":    f"{r['hospitalised']:,}",
                    "ICU":             f"{r['icu']:,}",
                    "Gen. Beds":       f"{r['beds_general']:,}",
                    "ICU Beds":        f"{r['beds_icu']:,}",
                    "Doctors":         f"{r['doctors_total']:,}",
                    "Nurses":          f"{r['nurses_total']:,}",
                    "PPE (Sets)":      f"{r['ppe_sets']:,}",
                    "Budget (₹ Cr)":   f"{r['total_budget_INR']/1e7:.1f}",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.divider()

            # Comparison charts
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(
                    _scenario_comparison_chart(scenarios, "infected", "Estimated Infected Cases"),
                    use_container_width=True,
                )
            with ch2:
                st.plotly_chart(
                    _scenario_comparison_chart(scenarios, "beds_general", "General Hospital Beds Needed"),
                    use_container_width=True,
                )

            ch3, ch4 = st.columns(2)
            with ch3:
                st.plotly_chart(
                    _scenario_comparison_chart(scenarios, "doctors_total", "Doctors Required"),
                    use_container_width=True,
                )
            with ch4:
                st.plotly_chart(
                    _scenario_comparison_chart(scenarios, "total_budget_INR", "Budget Estimate (INR)"),
                    use_container_width=True,
                )

            # Epidemic curves overlay
            st.divider()
            st.subheader("Epidemic Curve Overlay")
            fig_over = go.Figure()
            for sev, r in scenarios.items():
                d, daily, _ = epidemic_curve(r["infected"], r["horizon_days"], sev)
                fig_over.add_trace(go.Scatter(
                    x=d, y=daily, mode="lines", name=sev,
                    line=dict(color=SEVERITY_COLORS[sev], width=2),
                ))
            fig_over.update_layout(
                title="Daily New Cases by Severity Scenario",
                xaxis=dict(title="Day"),
                yaxis=dict(title="New Cases"),
                height=380,
                legend=dict(font=dict(color="#E6F1F5")),
                **PLOTLY_BASE,
            )
            st.plotly_chart(fig_over, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_export:
        if not result:
            _not_ready()
        else:
            st.subheader("Export Planning Report")
            st.divider()
            df_export = _result_to_df(result)
            st.dataframe(df_export, hide_index=True, use_container_width=True, height=500)
            st.divider()

            csv_bytes = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                    "Download as CSV",
                    data=csv_bytes,
                    file_name=f"resource_plan_{result['disease'].split('/')[0].strip().lower().replace(' ','_')}_{result['severity'].lower()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            st.divider()
