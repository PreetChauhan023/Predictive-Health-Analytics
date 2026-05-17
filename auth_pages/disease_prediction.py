import os
import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_DIR = os.path.join(_ROOT_DIR, "models_cache")

ALGO_OPTIONS = ["Random Forest", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes"]

SYMPTOM_CATEGORIES = {
    "General": ["fever", "fatigue", "chills", "sweating", "weight", "appetite",
                   "malaise", "lethargy", "weakness", "shivering", "restlessness",
                   "dehydration", "obesity", "excessive_hunger", "polyuria"],
    "Respiratory": ["cough", "breath", "throat", "phlegm", "sputum", "mucoid",
                       "sneez", "congestion", "runny", "sinus", "loss_of_smell"],
    "Cardiovascular": ["fast_heart", "chest_pain", "palpitation", "prominent_vein",
                          "swollen_legs", "swollen_blood", "fluid_overload",
                          "swelling_of_stomach", "distention", "swollen_extremeties"],
    "Neurological": ["headache", "dizziness", "anxiety", "depression", "mood",
                        "confusion", "altered", "balance", "spinning", "stiff_neck",
                        "paralysis", "slurred", "unsteadiness", "lack_of_concentration",
                        "irritability", "visual_disturbances", "blurred", "coma"],
    "Musculoskeletal": ["joint_pain", "muscle", "back_pain", "neck_pain", "knee",
                           "hip", "cramp", "stiff", "walking", "limb", "movement",
                           "pain_behind", "brittle_nails", "small_dents",
                           "inflammatory_nails", "swelling_joints"],
    "Skin": ["itching", "skin_rash", "nodal", "blister", "pimple", "blackhead",
                "scurring", "silver", "peeling", "eruption", "patch", "spots",
                "bruising", "dischromic", "red_spot", "yellow_crust", "red_sore"],
    "Digestive": ["stomach", "vomit", "nausea", "diarrhoea", "abdomen", "belly",
                     "constipation", "acidity", "ulcer", "bowel", "stool",
                     "passage_of_gas", "indigestion", "internal_itching",
                     "toxic_look", "bloody_stool", "irritation_in_anus",
                     "pain_during_bowel", "pain_in_anal"],
    "Liver / Jaundice": ["yellowish", "dark_urine", "yellowing_of_eyes",
                             "acute_liver", "stomach_bleeding", "history_of_alcohol",
                             "receiving_blood", "receiving_unsterile"],
    "Metabolic / Endocrine": ["irregular_sugar", "thyroid", "enlarged_thyroid",
                                  "abnormal_menstruation", "drying_and_tingling",
                                  "cold_hands_and_feets", "puffy_face"],
    "Eyes / ENT": ["watering_from_eyes", "sunken_eyes", "yellowing_of_eyes",
                       "patches_in_throat", "ulcers_on_tongue"],
}

RISK_LEVEL = {
    "Fungal infection": "🟢 Low", "Allergy": "🟢 Low", "Common Cold": "🟢 Low", "Acne": "🟢 Low",
    "GERD": "🟡 Medium", "Drug Reaction": "🟡 Medium", "Peptic ulcer diseae": "🟡 Medium",
    "Gastroenteritis": "🟡 Medium", "Bronchial Asthma": "🟡 Medium", "Migraine": "🟡 Medium",
    "Cervical spondylosis": "🟡 Medium", "Chicken pox": "🟡 Medium",
    "Dimorphic hemmorhoids(piles)": "🟡 Medium", "Varicose veins": "🟡 Medium",
    "Hypothyroidism": "🟡 Medium", "Hyperthyroidism": "🟡 Medium",
    "Osteoarthristis": "🟡 Medium", "Arthritis": "🟡 Medium",
    "(vertigo) Paroymsal  Positional Vertigo": "🟡 Medium",
    "Urinary tract infection": "🟡 Medium", "Psoriasis": "🟡 Medium", "Impetigo": "🟡 Medium",
    "hepatitis A": "🟡 Medium",
    "Chronic cholestasis": "🔴 High", "AIDS": "🔴 High", "Diabetes ": "🔴 High",
    "Hypertension ": "🔴 High", "Paralysis (brain hemorrhage)": "🔴 High",
    "Jaundice": "🔴 High", "Malaria": "🔴 High", "Dengue": "🔴 High",
    "Typhoid": "🔴 High", "Hepatitis B": "🔴 High", "Hepatitis C": "🔴 High",
    "Hepatitis D": "🔴 High", "Hepatitis E": "🔴 High",
    "Alcoholic hepatitis": "🔴 High", "Tuberculosis": "🔴 High",
    "Pneumonia": "🔴 High", "Heart attack": "🔴 High", "Hypoglycemia": "🔴 High",
}

PRECAUTIONS = {
    "Fungal infection":         ["Keep skin clean and dry", "Apply antifungal cream", "Avoid sharing personal items", "Wear breathable clothing"],
    "Allergy":                  ["Avoid known allergens", "Take antihistamines as prescribed", "Use air purifiers indoors", "Keep windows closed during high pollen season"],
    "GERD":                     ["Avoid spicy and fatty foods", "Eat smaller meals", "Don't lie down right after eating", "Elevate head while sleeping"],
    "Chronic cholestasis":      ["Consult a gastroenterologist", "Avoid alcohol completely", "Follow prescribed diet", "Get regular liver function tests"],
    "Drug Reaction":            ["Stop the suspected medication", "Consult your doctor immediately", "Carry an allergy card", "Report reaction to your physician"],
    "Peptic ulcer diseae":      ["Avoid spicy food and alcohol", "Take prescribed antacids", "Avoid NSAIDs like ibuprofen", "Reduce stress levels"],
    "AIDS":                     ["Follow antiretroviral therapy", "Practice safe sex", "Get regular CD4 count monitoring", "Consult an infectious disease specialist"],
    "Diabetes ":                ["Monitor blood glucose daily", "Follow a diabetic diet", "Exercise regularly", "Take prescribed medications consistently"],
    "Gastroenteritis":          ["Stay hydrated with ORS", "Eat bland foods like rice and toast", "Wash hands frequently", "Rest and avoid dairy products"],
    "Bronchial Asthma":         ["Always carry a rescue inhaler", "Avoid dust and smoke triggers", "Monitor peak flow regularly", "Take controller medications as prescribed"],
    "Hypertension ":            ["Reduce salt intake", "Exercise 30 minutes daily", "Monitor blood pressure regularly", "Take antihypertensives as prescribed"],
    "Migraine":                 ["Identify and avoid personal triggers", "Rest in a dark quiet room during attacks", "Take prescribed migraine medication early", "Maintain regular sleep schedule"],
    "Cervical spondylosis":     ["Do physiotherapy exercises daily", "Avoid prolonged screen time", "Use an ergonomic workstation", "Apply heat or cold packs to neck"],
    "Paralysis (brain hemorrhage)": ["Call emergency services immediately (112)", "Do not give food or water", "Keep the patient still and calm", "Immediate CT scan is required"],
    "Jaundice":                 ["Consult a hepatologist", "Avoid alcohol completely", "Stay well hydrated", "Follow a low-fat diet"],
    "Malaria":                  ["Take prescribed antimalarial drugs fully", "Use mosquito nets and repellents", "Eliminate stagnant water near home", "Seek immediate medical attention"],
    "Chicken pox":              ["Isolate to prevent spreading", "Avoid scratching blisters", "Use calamine lotion for relief", "Take antivirals if prescribed early"],
    "Dengue":                   ["Stay hydrated with plenty of fluids", "Monitor platelet count daily", "Avoid aspirin and NSAIDs", "Use mosquito repellents and eliminate breeding sites"],
    "Typhoid":                  ["Complete the full antibiotic course", "Drink only boiled or bottled water", "Eat freshly cooked food only", "Get vaccinated against typhoid"],
    "hepatitis A":              ["Rest and stay hydrated", "Avoid alcohol", "Eat healthy low-fat meals", "Vaccination recommended for prevention"],
    "Hepatitis B":              ["Get vaccinated if not already", "Avoid sharing needles or syringes", "Practice safe sex", "Get regular liver function monitoring"],
    "Hepatitis C":              ["Avoid sharing needles", "Get tested regularly if at risk", "Take prescribed antiviral therapy", "Avoid alcohol completely"],
    "Hepatitis D":              ["Hepatitis B vaccination prevents Hepatitis D", "Avoid blood-to-blood contact", "Regular liver monitoring", "Follow your hepatologist's advice"],
    "Hepatitis E":              ["Drink boiled or purified water", "Maintain proper sanitation", "Avoid raw or undercooked meat", "Rest and stay hydrated"],
    "Alcoholic hepatitis":      ["Stop alcohol consumption immediately", "Consult a hepatologist", "Follow prescribed nutritional diet", "Monitor liver enzymes regularly"],
    "Tuberculosis":             ["Complete the full TB medication course", "Cover mouth when coughing", "Ensure good ventilation at home", "Test close contacts regularly"],
    "Common Cold":              ["Rest and stay hydrated", "Use saline nasal drops", "Wash hands frequently", "Avoid close contact with others"],
    "Pneumonia":                ["Seek immediate medical care", "Complete prescribed antibiotics", "Rest and stay well hydrated", "Consider pneumococcal vaccination"],
    "Dimorphic hemmorhoids(piles)": ["Increase dietary fiber intake", "Stay well hydrated", "Avoid straining during bowel movements", "Warm sitz baths provide relief"],
    "Heart attack":             ["Call emergency services immediately (112)", "Chew aspirin if not allergic", "Keep the patient calm and still", "Do not leave the patient alone"],
    "Varicose veins":           ["Elevate legs when resting", "Wear compression stockings", "Exercise regularly", "Avoid prolonged standing or sitting"],
    "Hypothyroidism":           ["Take prescribed thyroid hormone replacement", "Get regular TSH blood tests", "Maintain a healthy diet", "Exercise regularly"],
    "Hyperthyroidism":          ["Take prescribed antithyroid medication", "Monitor heart rate", "Avoid excessive iodine", "Get regular endocrinologist check-ups"],
    "Hypoglycemia":             ["Eat regular meals without skipping", "Carry fast-acting glucose (candy/juice)", "Monitor blood sugar frequently", "Inform family or colleagues of condition"],
    "Osteoarthristis":          ["Do physiotherapy exercises regularly", "Maintain a healthy weight", "Use assistive devices if needed", "Take prescribed pain medication"],
    "Arthritis":                ["Do regular low-impact exercise", "Take anti-inflammatory medication as prescribed", "Apply hot or cold packs", "Protect joints during daily activities"],
    "(vertigo) Paroymsal  Positional Vertigo": ["Perform Epley maneuver as advised", "Move slowly and deliberately", "Avoid sudden head movements", "Consult an ENT specialist"],
    "Acne":                     ["Wash face twice daily with gentle cleanser", "Avoid touching your face", "Use non-comedogenic skincare products", "Consult a dermatologist for persistent acne"],
    "Urinary tract infection":  ["Drink plenty of water daily", "Complete the full antibiotic course", "Urinate after intercourse", "Maintain good personal hygiene"],
    "Psoriasis":                ["Moisturize skin regularly", "Avoid skin trauma and infections", "Take prescribed topical or systemic treatment", "Manage stress effectively"],
    "Impetigo":                 ["Keep the affected area clean", "Apply prescribed antibiotic cream", "Avoid touching or scratching sores", "Wash towels and clothing separately"],
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt(symptom: str) -> str:
    return symptom.replace("_", " ").strip().title()


def categorize_symptoms(feature_cols: list) -> dict:
    assigned = set()
    result = {cat: [] for cat in SYMPTOM_CATEGORIES}
    result["Other"] = []

    for cat, keywords in SYMPTOM_CATEGORIES.items():
        for sym in feature_cols:
            if sym not in assigned and any(kw in sym for kw in keywords):
                result[cat].append(sym)
                assigned.add(sym)

    for sym in feature_cols:
        if sym not in assigned:
            result["Other"].append(sym)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# DATA & MODEL  (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir    = os.path.abspath(os.path.join(current_dir, ".."))

    train_df = pd.read_csv(os.path.join(root_dir, "Training.csv"))
    test_df  = pd.read_csv(os.path.join(root_dir, "Testing.csv"))

    train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
    test_df  = test_df.loc[:, ~test_df.columns.str.contains("^Unnamed")]

    train_df["prognosis"] = train_df["prognosis"].str.strip()
    test_df["prognosis"]  = test_df["prognosis"].str.strip()

    feature_cols = [c for c in train_df.columns if c != "prognosis"]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["prognosis"]
    X_test  = test_df[feature_cols].reindex(columns=feature_cols, fill_value=0)
    y_test  = test_df["prognosis"]

    return X_train, y_train, X_test, y_test, feature_cols


@st.cache_resource(show_spinner="Loading models…")
def train_all_models():
    """
    Train all classifiers once and persist them to disk with joblib.
    On subsequent runs the saved files are loaded directly — no retraining.
    """
    os.makedirs(_MODELS_DIR, exist_ok=True)

    _SLUG = {
        "Random Forest":       "random_forest",
        "Decision Tree":       "decision_tree",
        "K-Nearest Neighbors": "knn",
        "Naive Bayes":         "naive_bayes",
    }
    _META_PATH = os.path.join(_MODELS_DIR, "disease_meta.joblib")

    # ── Try loading from disk first ──────────────────────────────────────────
    if os.path.exists(_META_PATH):
        try:
            meta    = joblib.load(_META_PATH)
            trained = {}
            for name, slug in _SLUG.items():
                model_path = os.path.join(_MODELS_DIR, f"disease_{slug}.joblib")
                trained[name] = {
                    "model":    joblib.load(model_path),
                    "accuracy": meta[name],
                }
            return trained
        except Exception:
            pass  # Loading failed — fall through and retrain

    # ── Train fresh and save ─────────────────────────────────────────────────
    X_train, y_train, X_test, y_test, _ = load_data()

    classifiers = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
    }

    trained = {}
    meta    = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        trained[name] = {"model": clf, "accuracy": round(acc * 100, 2)}
        meta[name]    = round(acc * 100, 2)
        joblib.dump(clf, os.path.join(_MODELS_DIR, f"disease_{_SLUG[name]}.joblib"))

    joblib.dump(meta, _META_PATH)
    return trained


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def page_disease_prediction():
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    all_models = train_all_models()
    categories = categorize_symptoms(feature_cols)
    st.title("Select ML algorithm")
    algo = st.selectbox("ML Algorithm", ALGO_OPTIONS)
    st.divider()
    # ── Symptom selection ─────────────────────────────────────────────────────
    st.subheader("Select Your Symptoms")
    all_selected = []
    for cat_name, sym_list in categories.items():
        if not sym_list:
            continue
        display_opts = [fmt(s) for s in sym_list]
        display_to_col = {fmt(s): s for s in sym_list}
        with st.expander(f"{cat_name}  —  {len(sym_list)} symptoms"):
            chosen = st.pills(
                label="",
                options=display_opts,
                selection_mode="multi",
                key=f"cat_{cat_name}"
            )
            if chosen:
                for d in chosen:
                    col = display_to_col.get(d)
                    if col:
                        all_selected.append(col)

    st.divider()

    # ── Selected summary ──────────────────────────────────────────────────────
    if all_selected:
        st.success(f"**{len(all_selected)} symptom(s) selected:** {', '.join(fmt(s) for s in all_selected)}")
    else:
        st.info("No symptoms selected yet. Open a category above to begin.")

    st.divider()

    # ── Predict button ────────────────────────────────────────────────────────
    if st.button("Predict Disease", type="primary", use_container_width=True):


        if not all_selected:
            st.error("Please select at least one symptom before prediction.")
            st.stop()

        clf = all_models[algo]["model"]

        input_vec = pd.DataFrame(
            [[1 if c in all_selected else 0 for c in feature_cols]],
            columns=feature_cols
        )

        probs = clf.predict_proba(input_vec)[0]
        classes = clf.classes_
        results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

        top_disease, top_prob = results[0]
        risk  = RISK_LEVEL.get(top_disease, "🟡 Medium")
        precs = PRECAUTIONS.get(top_disease, [])

        # ── Results ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Prediction Results")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Disease", top_disease.strip())
        col_b.metric("Confidence", f"{top_prob * 100:.1f}%")
        col_c.metric("Risk Level", risk)

        st.divider()

        # Top 5 table
        st.subheader("Top 5 Possible Diseases")
        table_data = {
            "Rank":       [f"#{i+1}" for i in range(5)],
            "Disease":    [r[0].strip() for r in results[:5]],
            "Confidence": [f"{r[1]*100:.1f}%" for r in results[:5]],
            "Risk":       [RISK_LEVEL.get(r[0], "🟡 Medium") for r in results[:5]],
        }

        st.dataframe(
            pd.DataFrame(table_data),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.warning("This is ML based Prediction. Please consult a qualified doctor.")