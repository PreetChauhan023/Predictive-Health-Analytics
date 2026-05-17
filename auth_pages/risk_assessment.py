import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, classification_report
import shap

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET     = os.path.join(BASE_DIR, "final_dataset.csv")
_MODELS_DIR = os.path.join(BASE_DIR, "models_cache")
_RISK_MODEL_PATH = os.path.join(_MODELS_DIR, "risk_model.joblib")

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_THRESHOLD = 33.0          # native Low/Medium boundary in dataset
RISK_COLOR     = {"Low": "#2ECC71", "Medium": "#F39C12"}
PLOTLY_BASE    = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6F1F5"),
)

CATEGORICAL = [
    'Gender', 'Smoking_Status', 'Alcohol_Consumption',
    'Physical_Activity_Level', 'Stress_Level', 'Diet_Type',
    'Region_Type', 'Vaccination_Status', 'State',
]
EXCLUDE = [
    'Patient_ID', 'Disease_Reported', 'Outbreak_Zone', 'Blood_Group',
    'Record_Date', 'Risk_Classification',
]

# Questionnaire options
STATES = sorted([
    "Andhra Pradesh","Assam","Bihar","Chhattisgarh","Delhi","Gujarat",
    "Haryana","Himachal Pradesh","Jharkhand","Karnataka","Kerala",
    "Madhya Pradesh","Maharashtra","Odisha","Punjab","Rajasthan",
    "Tamil Nadu","Telangana","Uttar Pradesh","Uttarakhand","West Bengal",
])

# ── Risk history storage ───────────────────────────────────────────────────────
RISK_HISTORY_DIR = os.path.join(BASE_DIR, "risk_history")

# ── Category sub-score definitions ────────────────────────────────────────────
CATEGORY_FEATURES = {
    "Vitals":          ["BMI", "Systolic_BP", "Diastolic_BP", "Heart_Rate",
                        "SpO2_Percent", "Blood_Glucose_mg_dL", "Cholesterol_mg_dL"],
    "Lifestyle":       ["Smoking_Status", "Alcohol_Consumption", "Physical_Activity_Level",
                        "Sleep_Hours", "Stress_Level", "Diet_Type"],
    "Medical History": ["Diabetes_History", "Hypertension_History", "Heart_Disease_History",
                        "Family_History_Chronic_Disease", "Previous_Hospitalizations"],
    "Environmental":   ["Air_Quality_Index", "Water_Quality_Score", "Region_Type",
                        "Vaccination_Status"],
}
CATEGORY_COLOR = {
    "Vitals":          "#3498DB",
    "Lifestyle":       "#F39C12",
    "Medical History": "#E74C3C",
    "Environmental":   "#2ECC71",
}

# ── Specialist recommendation lookup ──────────────────────────────────────────
SPECIALIST_MAP = {
    "Smoking":      ("Pulmonologist",
                     "Smoking-related lung and respiratory risk",
                     "📋 Request a spirometry test and lung health check-up."),
    "Activity":     ("Physiotherapist / Sports Medicine",
                     "Low activity & fitness level",
                     "📋 Ask for a personalised exercise plan tailored to your fitness."),
    "BMI_high":     ("Dietitian / Endocrinologist",
                     "BMI above healthy range",
                     "📋 Request a metabolic panel and dietary assessment."),
    "Hypertension": ("Cardiologist",
                     "Hypertension history",
                     "📋 Monitor BP weekly; ask for an ECG and lipid profile."),
    "Diabetes":     ("Endocrinologist",
                     "Diabetes history / elevated glucose",
                     "📋 HbA1c test every 3 months. Review carb intake with a dietitian."),
    "Alcohol":      ("Hepatologist / General Physician",
                     "Heavy alcohol consumption",
                     "📋 Request liver function tests (LFT) at your next visit."),
    "Stress":       ("Psychiatrist / Psychologist",
                     "High stress / mental health risk",
                     "📋 Consider a mental health screening and stress management plan."),
    "Sleep":        ("Sleep Specialist / Neurologist",
                     "Poor sleep quality",
                     "📋 Ask about a sleep study (polysomnography) if sleep is disrupted."),
}

# ── Tip key → feature importance key (for priority ranking) ───────────────────
TIP_FEATURE_MAP = {
    "Smoking":      "Smoking_Status",
    "Activity":     "Physical_Activity_Level",
    "BMI_high":     "BMI",
    "Hypertension": "Hypertension_History",
    "Diabetes":     "Diabetes_History",
    "Alcohol":      "Alcohol_Consumption",
    "Stress":       "Stress_Level",
    "Sleep":        "Sleep_Hours",
}

TIPS = {
    "Smoking":      ("🚬 Smoking is your top risk factor",
                     "Quitting or reducing smoking can lower your risk score by up to 15 points."),
    "Activity":     ("🏃 Low physical activity raises risk",
                     "Even 30 min of walking 5x per week significantly reduces risk."),
    "BMI_high":     ("⚖️ BMI above healthy range",
                     "Aim for BMI 18.5–24.9. Reducing 5kg can lower your score meaningfully."),
    "Hypertension": ("💉 Hypertension detected",
                     "Monitor BP monthly. Reduce salt, exercise regularly."),
    "Diabetes":     ("🩸 Diabetes history",
                     "Monitor blood glucose. Reduce refined carbs and sugar."),
    "Alcohol":      ("🍺 Heavy alcohol consumption",
                     "Aim for no more than 1–2 drinks/day. Include alcohol-free days."),
    "Stress":       ("🧠 High stress level",
                     "Mindfulness, yoga, or light exercise daily reduces cardiovascular risk."),
    "Sleep":        ("😴 Poor sleep",
                     "Aim for 7–8 hrs/night. Consistent sleep schedule helps significantly."),
    "Healthy":      ("Your profile looks healthy",
                     "Keep maintaining your current lifestyle. Annual checkups recommended."),
}


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train():
    """
    Train the risk-score RandomForestRegressor once and persist it to disk.
    On subsequent runs the saved model + encoders are loaded from disk directly.
    """
    os.makedirs(_MODELS_DIR, exist_ok=True)

    # ── Try loading from disk first ──────────────────────────────────────────
    if os.path.exists(_RISK_MODEL_PATH):
        try:
            return joblib.load(_RISK_MODEL_PATH)
        except Exception:
            pass  # Corrupt file — fall through and retrain

    # ── Train fresh ──────────────────────────────────────────────────────────
    df = pd.read_csv(DATASET)
    df['Alcohol_Consumption'] = df['Alcohol_Consumption'].fillna('Occasional')
    df['Record_Date']         = pd.to_datetime(df['Record_Date'], errors='coerce')
    df['Month']               = df['Record_Date'].dt.month.fillna(0).astype(int)

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in EXCLUDE + ['Risk_Score']]
    X = df[num_cols].fillna(0).copy()

    encoders = {}
    for col in CATEGORICAL:
        if col not in df.columns: continue
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = list(X.columns)
    y            = df['Risk_Score']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)

    preds     = model.predict(X_te)
    r2        = round(r2_score(y_te, preds), 4)
    mae       = round(mean_absolute_error(y_te, preds), 2)
    pop_mean  = round(float(y.mean()), 1)
    pop_std   = round(float(y.std()), 1)

    pred_labels = ['Low' if p <= RISK_THRESHOLD else 'Medium' for p in preds]
    true_labels = df.loc[y_te.index, 'Risk_Classification'].values
    label_acc   = round((np.array(pred_labels) == np.array(true_labels)).mean() * 100, 1)

    feature_importance = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda x: -x[1]
    )

    bundle = {
        "model":              model,
        "encoders":           encoders,
        "feature_cols":       feature_cols,
        "r2":                 r2,
        "mae":                mae,
        "label_accuracy":     label_acc,
        "pop_mean":           pop_mean,
        "pop_std":            pop_std,
        "feature_importance": feature_importance,
        "pop_scores":         y.values,
    }

    # ── Save to disk ─────────────────────────────────────────────────────────
    joblib.dump(bundle, _RISK_MODEL_PATH)
    return bundle


def predict(bundle, input_dict):
    model        = bundle["model"]
    encoders     = bundle["encoders"]
    feature_cols = bundle["feature_cols"]
    row = pd.DataFrame([input_dict])
    row['Month'] = pd.Timestamp.now().month
    for col, le in encoders.items():
        if col not in row.columns:
            row[col] = 0; continue
        val     = str(row[col].iloc[0]).strip()
        matches = [c for c in le.classes_ if c.lower() == val.lower()]
        row[col] = le.transform([matches[0] if matches else le.classes_[0]])[0]
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    score      = float(model.predict(row[feature_cols].fillna(0).values)[0])
    score      = round(np.clip(score, 0, 100), 1)
    risk_label = "Low" if score <= RISK_THRESHOLD else "Medium"

    pop_scores = bundle["pop_scores"]
    percentile = round(float((pop_scores < score).mean() * 100), 1)
    return score, risk_label, percentile



def _compute_shap_values(bundle, input_dict):
    model        = bundle["model"]
    encoders     = bundle["encoders"]
    feature_cols = bundle["feature_cols"]

    # Prepare row (same logic as predict)
    row = pd.DataFrame([input_dict])
    row['Month'] = pd.Timestamp.now().month

    for col, le in encoders.items():
        if col not in row.columns:
            row[col] = 0
            continue
        val = str(row[col].iloc[0]).strip()
        matches = [c for c in le.classes_ if c.lower() == val.lower()]
        row[col] = le.transform([matches[0] if matches else le.classes_[0]])[0]

    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0

    X_input = row[feature_cols].fillna(0)

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "impact": shap_values[0]
    })

    # Sort by absolute impact
    shap_df["abs_impact"] = shap_df["impact"].abs()
    shap_df = shap_df.sort_values(by="abs_impact", ascending=False)

    return shap_df.head(12)  # top features only




# ── Charts ────────────────────────────────────────────────────────────────────
def _gauge(score, risk_label):
    color = RISK_COLOR[risk_label]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100", "font": {"size": 38, "color": color}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#9FB4BE",
                      "tickvals": [0, 33, 100],
                      "ticktext": ["0", "33 (threshold)", "100"]},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,  33],  "color": "rgba(46,204,113,0.12)"},
                {"range": [33, 100], "color": "rgba(243,156,18,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 4},
                          "thickness": 0.75, "value": score},
        },
        title={"text": f"<b>Risk Score — {risk_label}</b>",
               "font": {"size": 18, "color": color}},
    ))
    fig.update_layout(height=290, margin=dict(t=40,b=10,l=20,r=20), **PLOTLY_BASE)
    return fig


def _population_chart(score, percentile, pop_scores):
    hist_vals, bin_edges = np.histogram(pop_scores, bins=30)
    bin_mids  = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    bar_colors = ["rgba(46,204,113,0.6)" if m <= RISK_THRESHOLD
                  else "rgba(243,156,18,0.6)" for m in bin_mids]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_mids, y=hist_vals,
        marker_color=bar_colors,
        hovertemplate="Score ~%{x:.1f}: %{y:,} patients<extra></extra>",
    ))
    fig.add_vline(
        x=score, line_color="#FFFFFF", line_width=2, line_dash="dash",
        annotation_text=f"  You ({score})",
        annotation_font_color="#FFFFFF", annotation_font_size=12,
    )
    fig.update_layout(
        title=f"Your score vs 30,000 patients — {percentile}th percentile",
        xaxis_title="Risk Score", yaxis_title="Patients",
        height=290, showlegend=False,
        margin=dict(t=50,b=40,l=40,r=20),
        **PLOTLY_BASE,
    )
    return fig


def _feature_chart(feature_importance):
    top    = feature_importance[:12]
    names  = [f.replace("_", " ").title() for f, _ in top]
    values = [round(float(v)*100, 2) for _, v in top]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=values, colorscale="Teal", showscale=False),
        text=[f"{v:.1f}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="What drives your risk score (model feature importance)",
        xaxis=dict(title="Importance (%)", range=[0, max(values)*1.3]),
        yaxis=dict(autorange="reversed"),
        height=420, margin=dict(t=40,b=10,l=160,r=60),
        **PLOTLY_BASE,
    )
    return fig



def _shap_chart(shap_df):
    df = shap_df.copy()
    df["feature"] = df["feature"].str.replace("_", " ").str.title()

    fig = go.Figure(go.Bar(
        x=df["impact"],
        y=df["feature"],
        orientation="h",
        text=[f"{v:+.2f}" for v in df["impact"]],
        textposition="outside",
        marker=dict(
            color=df["impact"],
            colorscale="RdBu",
            showscale=False
        ),
    ))

    fig.update_layout(
        title="What influenced your risk score (Personal Contribution)",
        xaxis_title="Impact on Risk Score",
        yaxis=dict(autorange="reversed"),
        height=420,
        margin=dict(t=40, b=10, l=160, r=60),
        **PLOTLY_BASE,
    )

    return fig


# ── Tips ──────────────────────────────────────────────────────────────────────
def _get_tips(inp):
    tips = []
    if inp.get("Smoking_Status") == "Current":      tips.append(TIPS["Smoking"])
    if inp.get("Physical_Activity_Level") == "Sedentary": tips.append(TIPS["Activity"])
    bmi = inp.get("BMI", 24)
    if bmi and float(bmi) >= 30:                    tips.append(TIPS["BMI_high"])
    if inp.get("Hypertension_History") == 1:        tips.append(TIPS["Hypertension"])
    if inp.get("Diabetes_History") == 1:            tips.append(TIPS["Diabetes"])
    if inp.get("Alcohol_Consumption") == "Heavy":   tips.append(TIPS["Alcohol"])
    if inp.get("Stress_Level") in ["High","Very High"]: tips.append(TIPS["Stress"])
    sleep = inp.get("Sleep_Hours", 7)
    if sleep and float(sleep) < 6:                  tips.append(TIPS["Sleep"])
    if not tips:                                     tips.append(TIPS["Healthy"])
    return tips




# ── NEW: Priority-ranked tips ─────────────────────────────────────────────────
def _get_tips_ranked(inp, feature_importance):
    """Return triggered tips sorted by model feature importance (highest first)."""
    importance_lookup = {feat: imp for feat, imp in feature_importance}

    triggered = []
    if inp.get("Smoking_Status") == "Current":
        triggered.append(("Smoking", TIPS["Smoking"]))
    if inp.get("Physical_Activity_Level") == "Sedentary":
        triggered.append(("Activity", TIPS["Activity"]))
    bmi = inp.get("BMI", 24)
    if bmi and float(bmi) >= 30:
        triggered.append(("BMI_high", TIPS["BMI_high"]))
    if inp.get("Hypertension_History") == 1:
        triggered.append(("Hypertension", TIPS["Hypertension"]))
    if inp.get("Diabetes_History") == 1:
        triggered.append(("Diabetes", TIPS["Diabetes"]))
    if inp.get("Alcohol_Consumption") == "Heavy":
        triggered.append(("Alcohol", TIPS["Alcohol"]))
    if inp.get("Stress_Level") in ["High", "Very High"]:
        triggered.append(("Stress", TIPS["Stress"]))
    sleep = inp.get("Sleep_Hours", 7)
    if sleep and float(sleep) < 6:
        triggered.append(("Sleep", TIPS["Sleep"]))

    if not triggered:
        return [("Healthy", TIPS["Healthy"])]

    # Sort by feature importance weight (descending)
    triggered.sort(
        key=lambda t: importance_lookup.get(TIP_FEATURE_MAP.get(t[0], ""), 0),
        reverse=True,
    )
    return triggered


# ── NEW: Category sub-scores ──────────────────────────────────────────────────
def _compute_category_scores(inp, feature_importance):
    """
    Compute a weighted risk sub-score (0–100) per category based on
    the feature importance of each feature within the category.
    """
    importance_lookup = {feat: imp for feat, imp in feature_importance}

    # Normalised risk contribution per feature (higher = worse)
    # Each value is mapped to 0–1 risk, then weighted by importance.
    RISK_SIGNAL = {
        # Vitals — higher deviations = more risk
        "BMI":                   lambda v: min(max((float(v) - 18.5) / 21.5, 0), 1),
        "Systolic_BP":           lambda v: min(max((float(v) - 90) / 110, 0), 1),
        "Diastolic_BP":          lambda v: min(max((float(v) - 60) / 70, 0), 1),
        "Heart_Rate":            lambda v: min(max((float(v) - 60) / 120, 0), 1),
        "SpO2_Percent":          lambda v: max(0, 1 - (float(v) - 70) / 30),
        "Blood_Glucose_mg_dL":   lambda v: min(max((float(v) - 70) / 330, 0), 1),
        "Cholesterol_mg_dL":     lambda v: min(max((float(v) - 100) / 300, 0), 1),
        # Lifestyle
        "Smoking_Status":        lambda v: {"Never": 0, "Former": 0.5, "Current": 1.0}.get(str(v), 0),
        "Alcohol_Consumption":   lambda v: {"Occasional": 0, "Moderate": 0.5, "Heavy": 1.0}.get(str(v), 0),
        "Physical_Activity_Level": lambda v: {"Active": 0, "Moderate": 0.33, "Light": 0.66, "Sedentary": 1.0}.get(str(v), 0),
        "Sleep_Hours":           lambda v: max(0, 1 - (float(v) / 8.0)),
        "Stress_Level":          lambda v: {"Low": 0, "Medium": 0.33, "High": 0.66, "Very High": 1.0}.get(str(v), 0),
        "Diet_Type":             lambda v: {"Vegan": 0, "Vegetarian": 0.2, "Non-Vegetarian": 0.5}.get(str(v), 0),
        # Medical history
        "Diabetes_History":      lambda v: float(v),
        "Hypertension_History":  lambda v: float(v),
        "Heart_Disease_History": lambda v: float(v),
        "Family_History_Chronic_Disease": lambda v: float(v),
        "Previous_Hospitalizations": lambda v: min(float(v) / 10, 1),
        # Environmental
        "Air_Quality_Index":     lambda v: min(max((float(v) - 20) / 480, 0), 1),
        "Water_Quality_Score":   lambda v: max(0, 1 - (float(v) / 100)),
        "Region_Type":           lambda v: {"Urban": 0.6, "Semi-Urban": 0.4, "Rural": 0.3}.get(str(v), 0.4),
        "Vaccination_Status":    lambda v: {"Fully Vaccinated": 0, "Partially Vaccinated": 0.5,
                                            "Not Vaccinated": 1.0}.get(str(v), 0),
    }

    scores = {}
    for category, features in CATEGORY_FEATURES.items():
        total_imp = sum(importance_lookup.get(f, 0.001) for f in features)
        weighted_risk = 0.0
        for feat in features:
            val = inp.get(feat, 0)
            imp = importance_lookup.get(feat, 0.001)
            sig = RISK_SIGNAL.get(feat, lambda v: 0)(val)
            weighted_risk += sig * (imp / total_imp)
        scores[category] = round(weighted_risk * 100, 1)

    return scores


# ── NEW: Radar chart ──────────────────────────────────────────────────────────
def _radar_chart(category_scores):
    categories = list(category_scores.keys())
    values     = [category_scores[c] for c in categories]
    colors     = [CATEGORY_COLOR[c] for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(52,152,219,0.15)",
        line=dict(color="#3498DB", width=2),
        mode="lines+markers",
        marker=dict(size=8, color=colors + [colors[0]]),
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[25, 50, 75, 100],
                gridcolor="rgba(255,255,255,0.15)",
                tickfont=dict(size=10),
            ),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title="Risk Breakdown by Category",
        height=380,
        margin=dict(t=60, b=20, l=60, r=60),
        **PLOTLY_BASE,
    )
    return fig


# ── NEW: Risk history helpers ─────────────────────────────────────────────────
def _save_risk_history(uid, score, risk_label, percentile):
    os.makedirs(RISK_HISTORY_DIR, exist_ok=True)
    path = os.path.join(RISK_HISTORY_DIR, f"{uid}.csv")
    record = pd.DataFrame([{
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "score":     score,
        "risk_label": risk_label,
        "percentile": percentile,
    }])
    if os.path.exists(path):
        record.to_csv(path, mode="a", header=False, index=False)
    else:
        record.to_csv(path, index=False)


def _load_risk_history(uid):
    path = os.path.join(RISK_HISTORY_DIR, f"{uid}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()


def _history_chart(history_df):
    fig = go.Figure()
    color_map = {"Low": "#2ECC71", "Medium": "#F39C12"}
    for label, grp in history_df.groupby("risk_label"):
        fig.add_trace(go.Scatter(
            x=grp["timestamp"], y=grp["score"],
            mode="markers",
            name=label,
            marker=dict(color=color_map.get(label, "#888"), size=8),
        ))
    fig.add_trace(go.Scatter(
        x=history_df["timestamp"], y=history_df["score"],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot"),
        showlegend=False,
    ))
    fig.add_hline(
        y=RISK_THRESHOLD, line_dash="dash",
        line_color="rgba(243,156,18,0.6)",
        annotation_text="Low / Medium threshold (33)",
        annotation_font_color="#E6F1F5",
    )
    fig.update_layout(
        title="Your Risk Score Over Time",
        xaxis_title="Date", yaxis_title="Risk Score",
        yaxis=dict(range=[0, 100]),
        height=320,
        margin=dict(t=50, b=40, l=50, r=20),
        legend=dict(orientation="h", y=1.1),
        **PLOTLY_BASE,
    )
    return fig

# ── Main page ─────────────────────────────────────────────────────────────────
def page_risk_assessment():
    st.title("Individual Risk Assessment")
    st.divider()
    uid = st.session_state.get("uid", "")
    if not uid:
        st.warning("Please log in to use the risk assessment.")
        return
    # Load + train model (cached)
    with st.spinner("Loading model..."):
        bundle = load_and_train()
    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("risk_form"):
        st.subheader("Fill in your details below and click Predict to see your risk score")
        # Row 1 — Demographics
        st.markdown("**Demographics**")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            age    = st.number_input("Age", min_value=1, max_value=89, value=30)
        with d2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with d3:
            state  = st.selectbox("State", STATES, index=STATES.index("Gujarat"))
        with d4:
            region = st.selectbox("Region Type", ["Urban", "Semi-Urban", "Rural"])
        st.divider()

        # Row 2 — Vitals (BMI via height/weight)
        st.markdown("**Vitals**")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=220, value=170)
            weight_kg = st.number_input("Weight (kg)", min_value=20,  max_value=200, value=70)
        with v2:
            systolic_bp   = st.number_input("Systolic BP (mmHg)", 80, 200, 120)
            diastolic_bp  = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)
        with v3:
            heart_rate    = st.number_input("Heart Rate (bpm)", 40, 180, 72)
            spo2          = st.number_input("SpO₂ (%)", 70, 100, 97)
        with v4:
            blood_glucose = st.number_input("Blood Glucose (mg/dL)", 50, 400, 100)
            cholesterol   = st.number_input("Cholesterol (mg/dL)", 100, 400, 190)

        st.divider()

        # Row 3 — Lifestyle
        st.markdown("**Lifestyle**")
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            smoking  = st.selectbox("Smoking Status",
                ["Never", "Former", "Current"])
            alcohol  = st.selectbox("Alcohol Consumption",
                ["Occasional", "Moderate", "Heavy"])
        with l2:
            activity = st.selectbox("Physical Activity Level",
                ["Active", "Moderate", "Light", "Sedentary"])
            diet     = st.selectbox("Diet Type",
                ["Vegetarian", "Non-Vegetarian", "Vegan"])
        with l3:
            sleep    = st.slider("Sleep Hours / night", 3.0, 12.0, 7.0, 0.5)
            stress   = st.selectbox("Stress Level",
                ["Low", "Medium", "High", "Very High"])
        with l4:
            vax      = st.selectbox("Vaccination Status",
                ["Fully Vaccinated", "Partially Vaccinated", "Not Vaccinated"])
            blood_grp= st.selectbox("Blood Group",
                ["A+","A-","B+","B-","O+","O-","AB+","AB-"])

        st.divider()

        # Row 4 — Medical history
        st.markdown("**Medical History**")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            diabetes     = st.radio("Diabetes History",
                ["No","Yes"], horizontal=True)
        with m2:
            hypertension = st.radio("Hypertension History",
                ["No","Yes"], horizontal=True)
        with m3:
            heart_dis    = st.radio("Heart Disease History",
                ["No","Yes"], horizontal=True)
        with m4:
            fam_history  = st.radio("Family History of Chronic Disease",
                ["No","Yes"], horizontal=True)

        prev_hosp = st.slider("Previous Hospitalizations", 0, 10, 0)

        st.divider()

        # Row 5 — Environmental
        st.markdown("**Environmental**")
        e1, e2 = st.columns(2)
        with e1:
            aqi           = st.slider("Air Quality Index", 20, 500, 150)
        with e2:
            water_quality = st.slider("Water Quality Score", 20, 100, 64)

        submitted = st.form_submit_button(
            "Predict My Risk Score", type="primary", use_container_width=True)

    # ── Run prediction ────────────────────────────────────────────────────────
    if submitted:
        # Calculate BMI
        h_m = height_cm / 100
        bmi = round(weight_kg / (h_m ** 2), 1) if h_m > 0 else 24.5

        input_dict = {
            "Age":                          int(age),
            "BMI":                          bmi,
            "Systolic_BP":                  int(systolic_bp),
            "Diastolic_BP":                 int(diastolic_bp),
            "Heart_Rate":                   int(heart_rate),
            "SpO2_Percent":                 float(spo2),
            "Blood_Glucose_mg_dL":          float(blood_glucose),
            "Cholesterol_mg_dL":            float(cholesterol),
            "Sleep_Hours":                  float(sleep),
            "Diabetes_History":             1 if diabetes == "Yes" else 0,
            "Hypertension_History":         1 if hypertension == "Yes" else 0,
            "Heart_Disease_History":        1 if heart_dis == "Yes" else 0,
            "Family_History_Chronic_Disease":1 if fam_history == "Yes" else 0,
            "Previous_Hospitalizations":    int(prev_hosp),
            "Air_Quality_Index":            int(aqi),
            "Water_Quality_Score":          int(water_quality),
            "Gender":                       gender,
            "Smoking_Status":               smoking,
            "Alcohol_Consumption":          alcohol,
            "Physical_Activity_Level":      activity,
            "Diet_Type":                    diet,
            "Stress_Level":                 stress,
            "Vaccination_Status":           vax,
            "State":                        state,
            "Region_Type":                  region,
            "Blood_Group":                  blood_grp,
        }

        score, risk_label, percentile = predict(bundle, input_dict)
        shap_df = _compute_shap_values(bundle, input_dict)
        # Save to history
        _save_risk_history(uid, score, risk_label, percentile)

        # Compute category sub-scores
        category_scores = _compute_category_scores(input_dict, bundle["feature_importance"])

        # Save to session for tabs
        st.session_state["_risk_result"] = {
            "score":            score,
            "risk_label":       risk_label,
            "percentile":       percentile,
            "input":            input_dict,
            "bmi":              bmi,
            "category_scores":  category_scores,
            "shap_df":          shap_df,
        }
        st.rerun()

    # ── Show results if available ─────────────────────────────────────────────
    if "_risk_result" not in st.session_state:
        return

    res             = st.session_state["_risk_result"]
    score           = res["score"]
    risk_label      = res["risk_label"]
    percentile      = res["percentile"]
    inp             = res["input"]
    bmi             = res["bmi"]
    category_scores = res.get("category_scores", {})
    color           = RISK_COLOR[risk_label]
    pop_mean        = bundle["pop_mean"]

    st.divider()

    # Banner
    diff     = round(score - pop_mean, 1)
    diff_str = f"{'+' if diff >= 0 else ''}{diff} vs population avg ({pop_mean})"
    st.container(border=True).markdown(
        f"<div style='text-align:center;padding:10px'>"
        f"<span style='font-size:20px;color:{color};font-weight:700'>"
        f"Risk Score: <u>{score} / 100</u> &nbsp;·&nbsp; "
        f"Level: {risk_label} &nbsp;·&nbsp; "
        f"{percentile}th percentile &nbsp;·&nbsp; {diff_str}"
        f"</span></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Score", "Analysis", "Recommendations",
        "Risk Breakdown", "Specialists", "History",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_gauge(score, risk_label), use_container_width=True)
        with c2:
            st.plotly_chart(
                _population_chart(score, percentile, bundle["pop_scores"]),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Key metrics from your input")
        bmi_lbl = "Normal" if 18.5 <= bmi < 25 else ("Overweight" if bmi >= 25 else "Underweight")
        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Risk Score",  f"{score} / 100")
        m2.metric("Risk Level",  risk_label)
        m3.metric("BMI",         f"{bmi}", bmi_lbl)
        m4.metric("Percentile",  f"{percentile}th")
        m5.metric("Smoking",     inp["Smoking_Status"])
        m6.metric("Activity",    inp["Physical_Activity_Level"])

    with tab2:
        st.subheader("Feature importance")
        st.caption("Which health factors the model considers most important for predicting risk.")
        st.plotly_chart(_feature_chart(bundle["feature_importance"]), use_container_width=True)

        st.divider()
        st.subheader("Model performance")
        p1,p2,p3 = st.columns(3)
        p1.metric("R² Score",       f"{bundle['r2']}")
        p2.metric("MAE",            f"±{bundle['mae']} pts")
        p3.metric("Label Accuracy", f"{bundle['label_accuracy']}%")
        st.caption(
            "RandomForestRegressor · 100 trees · trained on 30,000 records · "
            "Low ≤ 33 · Medium > 33 (native dataset threshold)"
        )
        st.divider()
        st.subheader("Personal Risk Explanation (SHAP)")

        shap_df = res.get("shap_df")

        if shap_df is not None:
            st.plotly_chart(_shap_chart(shap_df), use_container_width=True)

    with tab3:
        st.subheader("Recommendations — ranked by impact")
        st.caption("Tips are ordered by how much the model weights each factor. Address the top ones first.")
        st.write("")

        ranked_tips = _get_tips_ranked(inp, bundle["feature_importance"])
        for rank, (tip_key, (title, body)) in enumerate(ranked_tips, start=1):
            if tip_key == "Healthy":
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.write(body)
            else:
                priority_label = "🔴 High Priority" if rank == 1 else ("🟠 Medium Priority" if rank <= 3 else "🟡 Low Priority")
                with st.container(border=True):
                    col_a, col_b = st.columns([1, 8])
                    col_a.markdown(f"**#{rank}**")
                    col_b.markdown(f"{priority_label} &nbsp; **{title}**")
                    st.write(body)

        st.divider()
        if risk_label == "Medium":
            st.warning(
                f"Your risk score of **{score}** puts you in the **Medium** risk zone. "
                "Small lifestyle changes — especially around smoking, activity, and diet — "
                "can bring your score below 33."
            )
        else:
            st.success(
                f"Your risk score of **{score}** is in the **Low** risk zone. "
                "Keep maintaining your current lifestyle."
            )

    with tab4:
        st.subheader("Risk Breakdown by Category")
        st.caption("Each category score is a weighted risk estimate (0–100) based on your inputs and model feature importance.")

        if category_scores:
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.plotly_chart(_radar_chart(category_scores), use_container_width=True)
            with c_right:
                st.write("")
                st.write("")
                for cat, cat_score in category_scores.items():
                    cat_color = CATEGORY_COLOR[cat]
                    cat_label = "Low" if cat_score <= 33 else ("Medium" if cat_score <= 66 else "High")
                    st.markdown(
                        f"<div style='margin-bottom:10px'>"
                        f"<span style='font-weight:600;color:{cat_color}'>{cat}</span>"
                        f"<span style='float:right;color:{cat_color}'>{cat_score} / 100 · {cat_label}</span>"
                        f"</div>"
                        f"<div style='background:rgba(255,255,255,0.08);border-radius:6px;height:8px;margin-bottom:14px'>"
                        f"<div style='width:{cat_score}%;background:{cat_color};height:8px;border-radius:6px'></div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Submit the form to see your risk breakdown.")

    with tab5:
        st.subheader("Recommended Specialists")
        st.caption("Based on your triggered risk factors, these are the types of specialists you should consider consulting.")
        st.write("")

        ranked_tips = _get_tips_ranked(inp, bundle["feature_importance"])
        specialist_shown = set()
        any_shown = False

        for tip_key, _ in ranked_tips:
            if tip_key == "Healthy" or tip_key not in SPECIALIST_MAP:
                continue
            spec_name, reason, action = SPECIALIST_MAP[tip_key]
            if spec_name in specialist_shown:
                continue
            specialist_shown.add(spec_name)
            any_shown = True
            with st.container(border=True):
                s1, s2 = st.columns([2, 5])
                with s1:
                    st.markdown(f"#### 🏥 {spec_name}")
                with s2:
                    st.markdown(f"**Reason:** {reason}")
                    st.markdown(f"{action}")

        if not any_shown:
            st.success(
                "Your profile does not flag any specific specialist referrals. "
                "A general physician for an annual check-up is still recommended."
            )

    with tab6:
        st.subheader("Your Risk Score History")
        st.caption("Every time you run a prediction, the result is saved here so you can track your progress over time.")
        st.write("")

        history_df = _load_risk_history(uid)

        if history_df.empty:
            st.info("No history yet. Your first prediction has just been saved — run another one in a few days to see your trend.")
        else:
            st.plotly_chart(_history_chart(history_df), use_container_width=True)

            st.divider()
            st.markdown("**All assessments**")
            display_df = history_df.copy()
            display_df.columns = ["Date / Time", "Score", "Risk Level", "Percentile"]
            display_df = display_df.sort_values("Date / Time", ascending=False).reset_index(drop=True)
            st.dataframe(display_df, use_container_width=True, hide_index=True)