import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, mean_absolute_error,
                              r2_score, roc_auc_score, confusion_matrix)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════
#  Plotly theme — mirrors style.css colour variables
# ══════════════════════════════════════════════════════════
BG_MAIN    = "#0B1F2A"
BG_CARD    = "#112D3B"
BORDER     = "#2F4F4F"
PRIMARY    = "#2A9D8F"
TEXT_MAIN  = "#E6F1F5"
TEXT_MUTED = "#9FB4BE"
DANGER     = "#E76F51"
WARNING    = "#E9C46A"
SUCCESS    = "#2ECC71"

RISK_COLOR = {"Low": SUCCESS, "Medium": WARNING, "High": DANGER, "Critical": "#a855f7"}

def _plot(height=400):
    """Base Plotly layout matching style.css theme."""
    return dict(
        height=height,
        paper_bgcolor=BG_MAIN, plot_bgcolor=BG_CARD,
        font=dict(family="Inter, sans-serif", color=TEXT_MUTED, size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_MAIN)),
    )


# ══════════════════════════════════════════════════════════
#  ML PIPELINE  (trains once, then cached)
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading outbreak models…")
def load_and_train():
    import joblib

    BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH     = os.path.join(BASE_DIR, "outbreak.csv")
    CACHE_DIR    = os.path.join(BASE_DIR, "models_cache")
    BUNDLE_PATH  = os.path.join(CACHE_DIR, "outbreak_bundle.joblib")
    DF_PATH      = os.path.join(CACHE_DIR, "outbreak_df.joblib")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Try loading from disk first ───────────────────────────────────────────
    if os.path.exists(BUNDLE_PATH) and os.path.exists(DF_PATH):
        try:
            metrics = joblib.load(BUNDLE_PATH)
            df      = joblib.load(DF_PATH)
            return df, metrics
        except Exception:
            pass  # Corrupt cache — fall through and retrain

    # ── Train fresh ───────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])

    EXCLUDE  = ["Date", "Outbreak_Label", "Case_Count", "Risk_Level"]
    features = [c for c in df.columns if c not in EXCLUDE]
    X        = df[features].values
    split    = int(len(df) * 0.80)

    X_tr, X_te   = X[:split], X[split:]
    df_tr, df_te = df.iloc[:split], df.iloc[split:]

    y_cls_tr = df_tr["Outbreak_Label"].values;  y_cls_te = df_te["Outbreak_Label"].values
    y_reg_tr = df_tr["Case_Count"].values;       y_reg_te = df_te["Case_Count"].values

    le        = LabelEncoder()
    y_risk_tr = le.fit_transform(df_tr["Risk_Level"])
    y_risk_te = le.transform(df_te["Risk_Level"])

    # Model 1 — Outbreak Classifier
    clf = RandomForestClassifier(n_estimators=300, max_depth=12,
        min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_cls_tr)
    y_pred_cls = clf.predict(X_te)
    y_prob_cls = clf.predict_proba(X_te)[:, 1]

    # Model 2 — Case Count Regressor
    reg = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
        max_depth=5, subsample=0.8, random_state=42)
    reg.fit(X_tr, y_reg_tr)
    y_pred_reg = np.maximum(0, reg.predict(X_te)).round().astype(int)

    # Model 3 — Risk Level Classifier
    risk_clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
        max_depth=4, random_state=42)
    risk_clf.fit(X_tr, y_risk_tr)
    y_pred_risk = risk_clf.predict(X_te)

    # Full-dataset predictions for display
    df["Predicted_Cases"]      = np.maximum(0, reg.predict(X)).round().astype(int)
    df["Outbreak_Probability"] = np.round(clf.predict_proba(X)[:, 1], 4)
    df["Predicted_Outbreak"]   = clf.predict(X)
    df["Predicted_Risk"]       = le.inverse_transform(risk_clf.predict(X))
    roll30 = df["Predicted_Cases"].rolling(30, min_periods=1).mean()
    df["Alert_Flag"] = (
        (df["Outbreak_Probability"] > 0.50) |
        (df["Predicted_Cases"] > 1.75 * roll30)
    ).astype(int)

    metrics = {
        "clf_report" : classification_report(y_cls_te, y_pred_cls,
                        target_names=["No Outbreak","Outbreak"], zero_division=0, output_dict=True),
        "roc_auc"    : roc_auc_score(y_cls_te, y_prob_cls) if y_cls_te.sum() > 0 else None,
        "mae"        : mean_absolute_error(y_reg_te, y_pred_reg),
        "r2"         : r2_score(y_reg_te, y_pred_reg),
        "risk_report": classification_report(y_risk_te, y_pred_risk,
                        target_names=le.classes_, zero_division=0, output_dict=True),
        "conf_matrix": confusion_matrix(y_cls_te, y_pred_cls).tolist(),
        "fi_clf"     : pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False),
        "fi_reg"     : pd.Series(reg.feature_importances_,  index=features).sort_values(ascending=False),
        "features"   : features,
        "split_idx"  : split,
        "clf"        : clf,
        "reg"        : reg,
        "risk_clf"   : risk_clf,
        "le"         : le,
    }

    # ── Save to disk ──────────────────────────────────────────────────────────
    joblib.dump(metrics, BUNDLE_PATH)
    joblib.dump(df,      DF_PATH)

    return df, metrics



# ══════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════
def page_disease_outbreak_predictor():
    st.title("Disease Outbreak Forecasting")
    df, metrics = load_and_train()
    total_days    = len(df)
    total_alerts  = int(df["Alert_Flag"].sum())
    outbreak_days = int(df["Outbreak_Label"].sum())
    f1  = metrics["clf_report"]["Outbreak"]["f1-score"]
    r2  = metrics["r2"]
    # ── KPI strip ────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Days of Data",       total_days,    "2022–2024")
    k2.metric("Outbreak Days",      outbreak_days, f"{outbreak_days/total_days*100:.1f}% of total")
    k3.metric("Alerts Triggered",   total_alerts,  "by ML model")
    k4.metric("Classifier F1",      f"{f1:.3f}",   "Outbreak class")
    k5.metric("Regressor R²",       f"{r2:.4f}",   "Case count")
    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Outbreak Timeline",
        "Predict New Day",
        "Model Performance",
        "Feature Importance",
        "Alert Log",

    ])

    # ─────────────────────────────────────────────
    #  TAB 1 — Timeline
    # ─────────────────────────────────────────────
    with tab1:
        st.subheader("Case Count vs Predicted — Full Timeline")

        date_range = st.slider(
            "Date window",
            min_value=df["Date"].min().to_pydatetime(),
            max_value=df["Date"].max().to_pydatetime(),
            value=(df["Date"].min().to_pydatetime(), df["Date"].max().to_pydatetime()),
            format="YYYY-MM-DD",
        )
        mask = (df["Date"] >= date_range[0]) & (df["Date"] <= date_range[1])
        dv   = df[mask]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35], vertical_spacing=0.06)

        fig.add_trace(go.Scatter(x=dv["Date"], y=dv["Case_Count"],
            name="Actual Cases", line=dict(color=TEXT_MUTED, width=1.5),
            fill="tozeroy", fillcolor="rgba(159,180,190,0.08)"), row=1, col=1)

        fig.add_trace(go.Scatter(x=dv["Date"], y=dv["Predicted_Cases"],
            name="Predicted Cases", line=dict(color=PRIMARY, width=2, dash="dot")), row=1, col=1)

        ob = dv[dv["Outbreak_Label"] == 1]
        fig.add_trace(go.Scatter(x=ob["Date"], y=ob["Case_Count"],
            mode="markers", name="Outbreak Day",
            marker=dict(color=DANGER, size=8, line=dict(color=TEXT_MAIN, width=1))), row=1, col=1)

        fig.add_trace(go.Scatter(x=dv["Date"], y=dv["Outbreak_Probability"],
            name="Outbreak Probability", line=dict(color=WARNING, width=1.5),
            fill="tozeroy", fillcolor="rgba(233,196,106,0.10)"), row=2, col=1)

        fig.add_hline(y=0.5, line_dash="dash", line_color=DANGER, line_width=1, row=2, col=1)

        base = _plot(520)
        base.update({"yaxis":  dict(title="Cases",       gridcolor=BORDER, zeroline=False),
                     "yaxis2": dict(title="Probability", gridcolor=BORDER, zeroline=False, range=[0,1])})
        fig.update_layout(**base)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk Level Distribution")
        risk_counts = dv["Risk_Level"].value_counts().reindex(
            ["Critical","High","Medium","Low"], fill_value=0)
        fig2 = go.Figure(go.Bar(
            x=risk_counts.index, y=risk_counts.values,
            marker_color=[RISK_COLOR[r] for r in risk_counts.index],
            text=risk_counts.values, textposition="outside",
            textfont=dict(color=TEXT_MAIN),
        ))
        fig2.update_layout(**_plot(280))
        st.plotly_chart(fig2, use_container_width=True)

    # ─────────────────────────────────────────────
    #  TAB 2 — Predict New Day
    # ─────────────────────────────────────────────
    with tab2:
        st.subheader("Input Today's Health Parameters")
        st.info("Fill in the current day's values. The model will predict case count, outbreak probability, and risk level.")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Environmental**")
            temperature    = st.slider("Temperature (°C)",        10.0, 45.0, 28.0, 0.5)
            humidity       = st.slider("Humidity (%)",            20.0, 98.0, 65.0, 1.0)
            rainfall       = st.slider("Rainfall (mm)",            0.0, 80.0,  5.0, 0.5)
            ndvi           = st.slider("NDVI (Vegetation Index)", 0.05, 0.95,  0.4, 0.01)
            vector_density = st.slider("Vector Density (0–1)",     0.0,  1.0,  0.3, 0.01)
            livestock      = st.slider("Livestock Density",      200.0,900.0,500.0,10.0)

        with c2:
            st.markdown("**Health Infrastructure**")
            hosp_occ   = st.slider("Hospital Bed Occupancy (%)", 30.0, 99.0, 72.0, 1.0)
            vax        = st.slider("Vaccination Coverage (0–1)",  0.3,  0.98, 0.65, 0.01)
            med_supply = st.slider("Medical Supply Index (0–1)",  0.3,  1.0,  0.72, 0.01)
            sanitation = st.slider("Sanitation Index (0–1)",      0.2,  0.95, 0.60, 0.01)
            hw_ratio   = st.slider("Healthcare Workers/1000",     1.0,  5.0,  2.5,  0.1)
            st.markdown("**Mobility**")
            transit    = st.slider("Transit Usage (0–1)",         0.1,  1.0,  0.65, 0.01)
            travel     = st.slider("Travel Index (0–1)",          0.0,  1.0,  0.3,  0.01)
            gathering  = st.selectbox("Mass Gathering Event?", ["No", "Yes"])

        with c3:
            st.markdown("**Digital Signals**")
            search_vol   = st.slider("Search Volume (symptom hits)", 50, 300, 100, 5)
            sentiment    = st.slider("Social Media Panic (0–1)",  0.0, 1.0, 0.35, 0.01)
            clinic_surge = st.slider("Clinic Visit Surge",        0.4, 3.5,  1.0, 0.05)
            st.markdown("**Temporal / Lag**")
            month_val  = st.selectbox("Month", list(range(1, 13)), index=5)
            lag7       = st.number_input("Cases 7 days ago",   min_value=0, value=8)
            lag14      = st.number_input("Cases 14 days ago",  min_value=0, value=7)
            roll7_val  = st.number_input("7-day rolling avg",  min_value=0.0, value=8.0,  step=0.5)
            roll14_val = st.number_input("14-day rolling avg", min_value=0.0, value=7.5,  step=0.5)
            roll30_val = st.number_input("30-day rolling avg", min_value=0.0, value=7.0,  step=0.5)

        if st.button("Run Prediction", use_container_width=True, type="primary"):
            input_row = pd.DataFrame([{
                "Month": month_val, "Day_of_Year": month_val * 30,
                "Week_of_Year": month_val * 4,
                "Temperature": temperature, "Humidity": humidity, "Rainfall_mm": rainfall,
                "NDVI": ndvi, "Vector_Density": vector_density, "Livestock_Density": livestock,
                "Hosp_Bed_Occupancy": hosp_occ, "Vax_Coverage": vax,
                "Med_Supply_Index": med_supply, "Sanitation_Index": sanitation,
                "Healthcare_Worker_Ratio": hw_ratio, "Transit_Usage": transit,
                "Travel_Index": travel, "Mass_Gathering": 1 if gathering == "Yes" else 0,
                "Search_Volume": search_vol, "Social_Sentiment": sentiment,
                "Clinic_Visit_Surge": clinic_surge, "Cases_Lag_7": lag7,
                "Cases_Lag_14": lag14, "Cases_Roll_Avg_7": roll7_val,
                "Cases_Roll_Avg_14": roll14_val, "Cases_Roll_Avg_30": roll30_val,
                "Cases_Growth_Rate_7d":  (lag7  - roll14_val) / max(roll14_val, 1),
                "Cases_Growth_Rate_14d": (lag14 - roll30_val) / max(roll30_val, 1),
            }])[metrics["features"]]

            prob     = metrics["clf"].predict_proba(input_row)[0][1]
            outbreak = metrics["clf"].predict(input_row)[0]
            cases    = int(max(0, metrics["reg"].predict(input_row)[0]))
            risk     = metrics["le"].inverse_transform(
                           metrics["risk_clf"].predict(input_row))[0]

            st.divider()
            r1, r2c, r3, r4 = st.columns(4)
            r1.metric("Predicted Cases",       cases)
            r2c.metric("Outbreak Probability", f"{prob:.1%}")
            r3.metric("Outbreak Detected",     "⚠️ YES" if outbreak else "✅ NO")
            r4.metric("Risk Level",            risk)

            if outbreak or prob > 0.4:
                st.error(
                    f"**OUTBREAK ALERT** — Probability {prob:.1%} · "
                    f"Predicted {cases} cases · Risk: **{risk}**\n\n"
                    "Recommend immediate public health intervention and enhanced surveillance."
                )
            else:
                st.success(f"No outbreak detected. Probability: {prob:.1%}. Continue routine monitoring.")

    # ─────────────────────────────────────────────
    #  TAB 3 — Model Performance
    # ─────────────────────────────────────────────
    with tab3:
        st.subheader("Outbreak Classifier — Random Forest")
        rpt = metrics["clf_report"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Precision", f"{rpt['Outbreak']['precision']:.3f}")
        mc2.metric("Recall",    f"{rpt['Outbreak']['recall']:.3f}")
        mc3.metric("F1 Score",  f"{rpt['Outbreak']['f1-score']:.3f}")
        mc4.metric("ROC-AUC",   f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] else "N/A")

        st.subheader("Confusion Matrix")
        cm = np.array(metrics["conf_matrix"])
        fig_cm = px.imshow(cm, text_auto=True,
            x=["No Outbreak","Outbreak"], y=["No Outbreak","Outbreak"],
            color_continuous_scale=[[0, BG_CARD], [1, PRIMARY]],
            labels=dict(x="Predicted", y="Actual"))
        fig_cm.update_layout(**_plot(300))
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("Case Count Regressor — Gradient Boosting")
        mr1, mr2 = st.columns(2)
        mr1.metric("MAE (cases/day)", f"{metrics['mae']:.2f}")
        mr2.metric("R² Score",        f"{metrics['r2']:.4f}")

        st.subheader("Actual vs Predicted Cases — Test Set")
        test_df = df.iloc[metrics["split_idx"]:]
        fig_reg = go.Figure([
            go.Scatter(x=test_df["Date"], y=test_df["Case_Count"],
                name="Actual",    line=dict(color=TEXT_MUTED, width=1.5)),
            go.Scatter(x=test_df["Date"], y=test_df["Predicted_Cases"],
                name="Predicted", line=dict(color=PRIMARY, width=2, dash="dot")),
        ])
        fig_reg.update_layout(**_plot(300))
        st.plotly_chart(fig_reg, use_container_width=True)

        st.subheader("Risk Level Classifier — Performance")
        risk_rows = [
            {"Risk Level": lbl,
             "Precision": round(metrics["risk_report"][lbl]["precision"], 3),
             "Recall":    round(metrics["risk_report"][lbl]["recall"],    3),
             "F1":        round(metrics["risk_report"][lbl]["f1-score"],  3),
             "Support":   int(metrics["risk_report"][lbl]["support"])}
            for lbl in ["Low","Medium","High","Critical"]
            if lbl in metrics["risk_report"]
        ]
        st.dataframe(pd.DataFrame(risk_rows).set_index("Risk Level"), use_container_width=True)

    # ─────────────────────────────────────────────
    #  TAB 4 — Feature Importance
    # ─────────────────────────────────────────────
    with tab4:
        fi_choice = st.radio("Model", ["Outbreak Classifier", "Case Count Regressor"], horizontal=True)
        fi        = metrics["fi_clf"] if fi_choice == "Outbreak Classifier" else metrics["fi_reg"]
        top_n     = st.slider("Show top N features", 5, 31, 15)
        fi_top    = fi.head(top_n)

        fig_fi = go.Figure(go.Bar(
            x=fi_top.values[::-1], y=fi_top.index[::-1], orientation="h",
            marker=dict(color=fi_top.values[::-1],
                        colorscale=[[0, BG_CARD],[0.5, PRIMARY],[1, DANGER]],
                        showscale=False),
            text=[f"{v:.4f}" for v in fi_top.values[::-1]],
            textposition="outside", textfont=dict(color=TEXT_MUTED, size=10),
        ))
        base_fi = _plot(max(400, top_n * 28))
        base_fi["xaxis"]["title"] = "Importance"
        base_fi["margin"]["r"]    = 80
        fig_fi.update_layout(**base_fi)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.subheader("Correlation Heatmap — Top 12 Features")
        top12  = fi.head(12).index.tolist()
        corr   = df[top12 + ["Case_Count"]].corr()
        fig_hm = px.imshow(corr, text_auto=".2f",
                           color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        base_hm = _plot(480)
        base_hm["font"]["size"] = 9
        fig_hm.update_layout(**base_hm)
        st.plotly_chart(fig_hm, use_container_width=True)

    # ─────────────────────────────────────────────
    #  TAB 5 — Alert Log
    # ─────────────────────────────────────────────
    with tab5:
        st.subheader("Days with Outbreak Alerts")
        st.info(f"{total_alerts} alert days out of {total_days} total ({total_alerts/total_days*100:.1f}%)")

        alerts = df[df["Alert_Flag"] == 1][[
            "Date","Case_Count","Predicted_Cases",
            "Outbreak_Probability","Predicted_Risk","Outbreak_Label"
        ]].copy()
        alerts["Date"]                 = alerts["Date"].dt.strftime("%Y-%m-%d")
        alerts["Outbreak_Probability"] = (alerts["Outbreak_Probability"]*100).round(1).astype(str)+"%"
        alerts["Confirmed Outbreak"]   = alerts["Outbreak_Label"].map({1:"✅ Yes", 0:"— No"})
        alerts = alerts.drop(columns=["Outbreak_Label"])
        alerts.columns = ["Date","Actual Cases","Predicted Cases","Probability","Risk Level","Confirmed Outbreak"]
        st.dataframe(alerts, use_container_width=True, height=500)
