# 🏥 Medical Insurance Cost Prediction

Predict the yearly medical insurance **charges** of a person from simple details (age, sex, BMI, children, smoker, region) using machine learning, with a **Streamlit** web app to try it out.

## Results
| Model | R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.52 | $3,983 | $7,965 |
| Random Forest | 0.86 | $2,079 | $4,372 |
| **Gradient Boosting (best)** | **0.87** | **$1,982** | **$4,194** |

Key findings:
- **Smoking is the biggest cost driver**: smokers pay about 3.8x more on average ($32,050 vs $8,441).
- A higher BMI (especially obese, BMI 30 and above) means higher cost.
- Charges are right-skewed, so the model predicts `log(charges)` and converts back to dollars.

*(Scores can vary slightly between runs because of the random train/test split.)*

## Project structure
```
.
├── Medical_Insurance_Cost_Prediction.ipynb   # full analysis: cleaning, EDA, modelling, evaluation
├── app.py                                    # Streamlit web app
├── medical_insurance.csv                     # dataset
├── medical_insurance_model.pkl               # trained model (Gradient Boosting)
├── model_features.pkl                        # column names/order used for training
├── requirements.txt                          # Python packages
└── README.md
```

## Steps in the notebook
1. Load the data and check quality (missing values, duplicates)
2. Remove duplicate rows (2,772 -> 1,337 unique rows)
3. Exploratory data analysis (distributions, outliers, smoker and BMI effects)
4. Feature engineering (BMI category, log-transformed target)
5. One-hot encoding of text columns
6. Train/test split, then train and compare 3 models
7. Save the best model with `joblib`

## Run it locally
```bash
# 1. clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. install the packages
pip install -r requirements.txt

# 4. start the web app
streamlit run app.py
```
To re-train the model, open the notebook and use **Run All**.

## Tech stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit, joblib

## Limitations
This is a learning project. The dataset is small (1,337 people) and has only 6 inputs, so it must not be used to set real insurance prices.

## Dataset
Medical insurance cost data (the commonly used public "insurance.csv" dataset). Check the original source's license before reuse.
