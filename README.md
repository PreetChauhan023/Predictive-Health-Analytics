# Predictive Health Analytics

A machine learning-powered health intelligence platform built with Streamlit. It assesses individual health risks, forecasts disease outbreaks, interprets lab reports using AI, and provides personalized health recommendations.

Live Demo: [predictive-health-analytics.streamlit.app](https://preetchauhan023-predictive-health-analytics.streamlit.app)

---

## Features

| Module | Description |
|---|---|
| Risk Assessment | Personalized health risk scoring (0–100) with Low / Medium classification using Random Forest |
| Disease Predictor | Multi-algorithm disease prediction (Random Forest, Decision Tree, KNN, Naive Bayes) |
| Lab Report Interpreter | Upload a PDF or image lab report — AI extracts and explains every biomarker |
| Disease Outbreak Predictor | Regional outbreak forecasting using population-level trends and Gradient Boosting |
| AI Health Chatbot | LLM-powered medical assistant (Groq) for health questions |
| Dataset Analysis | Interactive exploration of the 30,000-record pan-India health dataset |
| Resource Planning | Healthcare resource allocation analytics |
| Consult a Doctor | Doctor consultation request system |

---

## Tech Stack

- **Frontend** — Streamlit, Plotly, Custom CSS
- **ML Models** — Scikit-learn (Random Forest, Gradient Boosting, KNN, Decision Tree, Naive Bayes), SHAP for explainability
- **AI APIs** — Groq (chatbot), Google Gemini (lab report interpretation)
- **PDF Processing** — PyMuPDF (fitz)
- **Data** — Pandas, NumPy, CSV-based storage

---

## Project Structure

```
predictive-health-analytics/
├── app.py                        # Main entry point & router
├── style.css                     # Custom theme & responsive styles
├── requirements.txt
│
├── auth_pages/
│   ├── login.py                  # Login page
│   ├── signup.py                 # Registration page
│   ├── otp_verify.py             # OTP verification
│   ├── dashboard.py              # Main dashboard & sidebar navigation
│   ├── home.py                   # Home page with KPI cards
│   ├── risk_assessment.py        # Risk scoring module
│   ├── disease_prediction.py     # Disease prediction module
│   ├── disease_outbreak.py       # Outbreak forecasting module
│   ├── labreport.py              # Lab report interpreter
│   ├── chatbot.py                # AI health chatbot
│   ├── dataset.py                # Dataset analysis
│   ├── resource.py               # Resource planning
│   ├── consult.py                # Doctor consultation
│   ├── profile.py                # User profile
│   ├── settings.py               # Account settings
│   └── about.py                  # About page
│
├── services/
│   └── interpreter.py            # Gemini + Groq lab report AI service
│
├── storage/
│   └── repository.py             # Lab report CSV storage
│
├── models/
│   └── schema.py                 # Gemini response JSON schema
│
├── utils/
│   ├── config.py                 # API key loader (Streamlit secrets)
│   ├── validators.py             # Email, phone, password validators
│   └── visualization.py         # Reusable chart components
│
└── data files (gitignored)
    ├── final_dataset.csv         # 30,000-record health dataset
    ├── Training.csv / Testing.csv
    ├── outbreak.csv
    └── user_registration_data.csv
```

---

## How It Works

1. **Data Collection** — User enters health parameters: vitals, lifestyle, medical history, demographics
2. **ML Processing** — Models trained on 30,000 records process inputs and generate predictions
3. **Risk Scoring** — A risk score (0–100) is assigned with Low or Medium classification
4. **Recommendations** — Personalized health advice and population context for preventive action

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/PreetChauhan023/Predictive-Health-Analytics.git
cd Predictive-Health-Analytics

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API keys
# Create .streamlit/secrets.toml and add:
# GEMINI_API_KEY = "your-key"
# GROQ_API_KEY = "your-key"

# 5. Run
streamlit run app.py
```

---

## Dataset

- 30,000+ patient records across 28 Indian states
- 41 disease categories
- Features include age, gender, BMI, blood pressure, lifestyle factors, medical history, and environmental data

---

## Developed by

Preet Chauhan
