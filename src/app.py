import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.rag_engine import MedicalRAGEngine
from src.prediction import predict_diabetes_risk


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MediSense AI",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 MediSense AI")

st.subheader(
    "AI-Powered Healthcare Assessment Assistant"
)

st.write(
    "Diabetes Risk Prediction and Evidence-Based Medical Q&A"
)


# ==========================================
# LOAD RAG ENGINE
# ==========================================

@st.cache_resource
def load_rag_engine():
    return MedicalRAGEngine()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("MediSense AI")

page = st.sidebar.radio(
    "Select a Feature",
    [
        "Diabetes Risk Prediction",
        "Medical Q&A Assistant"
    ]
)


# ==========================================
# FEATURE 1: DIABETES PREDICTION
# ==========================================

if page == "Diabetes Risk Prediction":

    st.header("Diabetes Risk Prediction")

    st.write(
        "Enter patient health information to generate "
        "machine-learning predictions."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        # First column
        with col1:

            st.subheader("Health Indicators")

            high_bp = st.selectbox(
                "High Blood Pressure", [0, 1]
            )

            high_chol = st.selectbox(
                "High Cholesterol", [0, 1]
            )

            chol_check = st.selectbox(
                "Cholesterol Check", [0, 1]
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=80.0,
                value=25.0
            )

            smoker = st.selectbox(
                "Smoker", [0, 1]
            )

            stroke = st.selectbox(
                "Previous Stroke", [0, 1]
            )

            heart_disease = st.selectbox(
                "Heart Disease or Heart Attack", [0, 1]
            )

            phys_activity = st.selectbox(
                "Physical Activity", [0, 1]
            )

            fruits = st.selectbox(
                "Consumes Fruits", [0, 1]
            )

            veggies = st.selectbox(
                "Consumes Vegetables", [0, 1]
            )

            alcohol = st.selectbox(
                "Heavy Alcohol Consumption", [0, 1]
            )

        # Second column
        with col2:

            st.subheader("Additional Information")

            healthcare = st.selectbox(
                "Has Healthcare Coverage", [0, 1]
            )

            no_doc_cost = st.selectbox(
                "Could Not See Doctor Due to Cost", [0, 1]
            )

            gen_health = st.slider(
                "General Health (1-5)", 1, 5, 3
            )

            ment_health = st.slider(
                "Mental Health Unhealthy Days", 0, 30, 0
            )

            phys_health = st.slider(
                "Physical Health Unhealthy Days", 0, 30, 0
            )

            diff_walk = st.selectbox(
                "Difficulty Walking", [0, 1]
            )

            sex = st.selectbox(
                "Sex (0 = Female, 1 = Male)", [0, 1]
            )

            age = st.slider(
                "Age Category (1-13)", 1, 13, 8
            )

            education = st.slider(
                "Education Level (1-6)", 1, 6, 4
            )

            income = st.slider(
                "Income Category (1-8)", 1, 8, 5
            )

        submitted = st.form_submit_button(
            "Predict Diabetes Risk"
        )

    # Generate predictions
    if submitted:

        patient_data = {

            "HighBP": high_bp,
            "HighChol": high_chol,
            "CholCheck": chol_check,
            "BMI": bmi,
            "Smoker": smoker,
            "Stroke": stroke,
            "HeartDiseaseorAttack": heart_disease,
            "PhysActivity": phys_activity,
            "Fruits": fruits,
            "Veggies": veggies,
            "HvyAlcoholConsump": alcohol,
            "AnyHealthcare": healthcare,
            "NoDocbcCost": no_doc_cost,
            "GenHlth": gen_health,
            "MentHlth": ment_health,
            "PhysHlth": phys_health,
            "DiffWalk": diff_walk,
            "Sex": sex,
            "Age": age,
            "Education": education,
            "Income": income

        }

        try:

            result = predict_diabetes_risk(patient_data)

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Random Forest Probability",
                    f"{result['Random Forest Probability']:.2f}%"
                )

                st.write(
                    "Prediction:",
                    "Higher-risk class"
                    if result["Random Forest Prediction"] == 1
                    else "Lower-risk class"
                )

            with col2:

                st.metric(
                    "Logistic Regression Probability",
                    f"{result['Logistic Regression Probability']:.2f}%"
                )

                st.write(
                    "Prediction:",
                    "Higher-risk class"
                    if result["Logistic Regression Prediction"] == 1
                    else "Lower-risk class"
                )

            st.warning(
                "These are experimental machine-learning outputs, "
                "not clinically validated risk estimates or diagnoses. "
                "Consult a qualified healthcare professional for "
                "medical assessment."
            )

        except Exception as error:

            st.error(f"Prediction failed: {error}")


# ==========================================
# FEATURE 2: MEDICAL Q&A ASSISTANT
# ==========================================

elif page == "Medical Q&A Assistant":

    st.header("Medical Q&A Assistant")

    st.write(
        "Ask questions about diabetes, hypertension, "
        "symptoms and lifestyle guidance."
    )

    question = st.text_area(
        "Enter your medical question",
        placeholder=(
            "Example: What are the common risk factors for diabetes?"
        )
    )

    if st.button("Get Answer"):

        if question.strip():

            with st.spinner(
                "Searching medical documents..."
            ):

                try:

                    rag = load_rag_engine()

                    response = rag.ask_question(question)

                    st.subheader("Medical Answer")

                    st.write(response["answer"])

                    st.subheader("Medical Document Sources")

                    if response["sources"]:

                        for source in response["sources"]:

                            st.write(
                                f"📄 Document: {source['source']}"
                            )

                            st.write(
                                f"Page: {source['page']}"
                            )

                            st.write(
                                f"Similarity Score: "
                                f"{source['similarity_score']}"
                            )

                            st.divider()

                    else:

                        st.info(
                            "No relevant document sources were retrieved."
                        )

                    st.info(
                        "This assistant provides educational information "
                        "based on retrieved medical documents. "
                        "It does not replace professional medical advice."
                    )

                except Exception as error:

                    st.error(
                        f"Unable to generate answer: {error}"
                    )

        else:

            st.warning(
                "Please enter a medical question."
            )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "MediSense AI | Machine Learning + "
    "Retrieval-Augmented Generation (RAG)"
)