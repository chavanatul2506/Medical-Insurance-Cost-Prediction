# =====================================================================
# Insurance Cost Estimator - Streamlit web app
#
# What this app does:
#   1. Loads the trained model (saved by the notebook with joblib).
#   2. Shows a form where the user enters age, sex, BMI, children,
#      smoker status and region.
#   3. Converts those inputs into the same numeric format the model was
#      trained on.
#   4. Predicts the yearly insurance charge and compares it with the
#      average charge of non-smokers and smokers.
#
# Run it with:  streamlit run Medical_Insurance_Cost_Prediction_app.py
# =====================================================================

# ---------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------
import os                      # to check whether the model files exist
import numpy as np             # np.expm1 converts log-charges back to dollars
import streamlit as st         # builds the web page
import pandas as pd            # tables (the one-row input for the model)
import joblib                  # loads the saved model files
import warnings
warnings.filterwarnings("ignore")   # hide warning messages to keep the page clean

# ---------------------------------------------------------------------
# 2. Page settings
# NOTE: st.set_page_config must be the FIRST Streamlit command in the script.
# "wide" layout uses the full width of the browser window.
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Cost Estimator",
    layout="wide"
)

# ---------------------------------------------------------------------
# 3. Settings and constants
# Full paths to the two files saved by the notebook:
#   - the trained Gradient Boosting model
#   - the list of input columns (in the exact order used for training)
# These are fixed paths on this computer. To run the app on another
# machine, put the .pkl files next to this script and use just the file
# names, for example "medical_insurance_model.pkl".
# ---------------------------------------------------------------------
MODEL_PATH =  "medical_insurance_model.pkl"
FEATURES_PATH = "model_features.pkl"

# Average yearly charge in the dataset, used as a comparison in the chart.
# (Found in the notebook: non-smokers about $8,441, smokers about $32,050.)
REF_AVG_NON_SMOKER = 8400
REF_AVG_SMOKER = 32000

# ---------------------------------------------------------------------
# 4. Load the model
# @st.cache_resource makes Streamlit load the files only ONCE and reuse
# them, instead of reading them again every time the user moves a slider.
# If a file is missing, the function returns (None, None) so the app can
# show a friendly message instead of crashing.
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    return model, feature_cols

# ---------------------------------------------------------------------
# 5. Helper: BMI category
# Same rules as in the notebook, so the app and the training data agree:
#   below 18.5 -> underweight | 18.5 to 24.9 -> normal
#   25 to 29.9 -> overweight  | 30 and above -> obese
# ---------------------------------------------------------------------
def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "overweight"
    else:
        return "obese"

# ---------------------------------------------------------------------
# 6. Helper: turn the user's inputs into ONE row for the model
# The model was trained on one-hot encoded columns (sex_male, smoker_yes,
# region_..., bmi_category_...), so we must create exactly the same
# columns here. The reference groups (female, non-smoker, northeast,
# normal BMI) have no column: they are represented by all 0s.
# ---------------------------------------------------------------------
def build_input_row(age, sex, bmi, children, smoker, region, feature_cols):
    # find the BMI group (underweight / normal / overweight / obese)
    category = bmi_category(bmi)

    # one dictionary entry per model input column.
    # "1 if ... else 0" is the manual one-hot encoding.
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        "bmi_category_obese": 1 if category == "obese" else 0,
        "bmi_category_overweight": 1 if category == "overweight" else 0,
        "bmi_category_underweight": 1 if category == "underweight" else 0,
    }
    # (this variable is not used further; the line below does the real work)
    encoded = pd.DataFrame([row])
    # make a one-row table and force the columns to match the training
    # columns: same names, same order (missing ones are filled with 0).
    return pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)

# ---------------------------------------------------------------------
# 7. Load the model and show the page title
# ---------------------------------------------------------------------
model, feature_cols = load_artifacts()
st.title("Insurance Cost Estimator")

# If the model files were not found, show an error and stop the script
# here, so nothing below runs and crashes.
if model is None:
    st.error("Model files not found.")
    st.stop()

# ---------------------------------------------------------------------
# 8. Page layout: two columns, inputs on the left, results on the right.
# The numbers [1, 1.2] are relative widths (the right side is 20% wider).
# ---------------------------------------------------------------------
left, right = st.columns([1, 1.2], gap="large")

# ---------------------------------------------------------------------
# 9. Left column: the input form
# Each widget returns the value the user chose. The option texts are
# lowercase because build_input_row compares against lowercase words.
# ---------------------------------------------------------------------
with left:
    st.subheader("Patient Details")

    age = st.slider("Age", 18, 64, 30)                                   # min 18, max 64, default 30
    sex = st.radio("Sex", ["male", "female"], horizontal=True)           # radio buttons side by side
    bmi = st.slider("BMI", 15.0, 55.0, 26.0, 0.1)                        # min, max, default, step
    children = st.slider("Children", 0, 5, 0)
    smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])  # drop-down

    # The button returns True only on the run right after it is clicked.
    predict_clicked = st.button("Estimate charges", type="primary", use_container_width=True)

# ---------------------------------------------------------------------
# 10. Right column: prediction and comparison chart
# ---------------------------------------------------------------------
with right:

    st.subheader("Estimated charges")

    if predict_clicked:
        # Step 1: build the one-row input in the training format
        input_row = build_input_row(age, sex, bmi, children, smoker, region, feature_cols)

        # Step 2: predict. The model was trained on log(charges), so its
        # output is a log value.
        log_pred = model.predict(input_row)[0]

        # Step 3: convert back to dollars (np.expm1 reverses np.log1p)
        pred_charges = np.expm1(log_pred)

        # Step 4: small table for the chart: the prediction next to the
        # average non-smoker and smoker charges
        compare_df = pd.DataFrame({
            "Group" : ["Your prediction", "Non-smoker avg.", "Smoker avg."],
                "Charges" : [pred_charges, REF_AVG_NON_SMOKER, REF_AVG_SMOKER]
                    }).set_index("Group")

        # Step 5: show the number (with commas and 2 decimals) and the bar chart
        st.metric("Predicted annual charge", f"${pred_charges:,.2f}")
        st.bar_chart(compare_df)

    else:
        # before the first click, show a hint instead of a result
        st.info("Fill in the details and click **Estimate charges**.")
