import joblib
import numpy as np
import pandas as pd
import os
from scipy.sparse import hstack, csr_matrix
print("Current Working Directory:", os.getcwd())
# ---------- Load artifacts (update paths as needed) ----------
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.joblib')   # or .pkl
onehot_encoder    = joblib.load('models/onehot_encoder.joblib')
xgb_model         = joblib.load('models/xgboost_model.joblib')

# ---------- Defaults for categorical/numeric values ----------
# These should match how you prepared defaults during training (e.g., filled NaNs with 'Unknown')
DEFAULT_EMPLOYMENT = 'Unknown'
DEFAULT_EXPERIENCE = 'Unknown'
DEFAULT_EDUCATION  = 'Unknown'
# Numeric defaults (telecommuting, has_company_logo, has_questions)
DEFAULT_NUMERIC = {'telecommuting': 0, 'has_company_logo': 0, 'has_questions': 0}

# ---------- Inference function ----------
def predict_job_posting(text,
                        model,
                        tfidf=tfidf_vectorizer,
                        encoder=onehot_encoder,
                        employment_type=DEFAULT_EMPLOYMENT,
                        required_experience=DEFAULT_EXPERIENCE,
                        required_education=DEFAULT_EDUCATION,
                        telecommuting=0,
                        has_company_logo=0,
                        has_questions=0,
                        verbose=True):
    """
    Build the combined feature vector (TF-IDF | OHE | numeric) and predict using the provided model.
    Returns: (prediction_label, proba_fake)
    """
    # 1) Text -> TF-IDF (sparse)
    text_vectorized = tfidf.transform([text])  # shape (1, n_tfidf_features), sparse csr

    # 2) Prepare categorical dataframe exactly like during training
    cat_df = pd.DataFrame([{
        'employment_type': employment_type,
        'required_experience': required_experience,
        'required_education': required_education
    }])

    # 3) Transform categorical -> sparse OHE
    # encoder was fitted on train with handle_unknown='ignore' and sparse_output=True
    cat_vectorized = encoder.transform(cat_df)   # sparse matrix (1, n_cat_features)

    # 4) Numeric features -> sparse row
    num_arr = np.array([[telecommuting, has_company_logo, has_questions]])  # shape (1,3)
    num_sparse = csr_matrix(num_arr)  # convert to sparse

    # 5) Combine horizontally: TF-IDF | OHE | numeric
    combined = hstack([text_vectorized, cat_vectorized, num_sparse], format='csr')  # sparse csr

    # 6) Some XGBoost wrappers accept sparse matrices directly; if your saved model complains, convert:
    try:
        proba = model.predict_proba(combined)    # (1,2)
        pred  = model.predict(combined)[0]
    except Exception as e:
        # fallback: convert to dense (only for a single row - cheap). Avoid dense for large batches.
        combined_dense = combined.toarray()
        proba = model.predict_proba(combined_dense)
        pred  = model.predict(combined_dense)[0]

    proba = np.asarray(proba).ravel()
    label = 'Fake' if int(pred) == 1 else 'Real'

    if verbose:
        print("Prediction:", label)
        print("Probability (Real): {:.4f}".format(proba[0]))
        print("Probability (Fake): {:.4f}".format(proba[1]))

    return label, proba[1]

# ---------- Example usage ----------
job_text = """
We are hiring for Multiple Positions in ATOS.

Eligible: College Students & Freshers 
 
Virtual Interview on 18th and 19th November 2025 | Timing:- 07:00 PM

Comment your Email Address for Apply 🔗 Link 

Eligibility Criteria 

Internship Salary 15-25 K & PPO Salary Range: 5 to 10 LPA
* Job Types: WFH/WFO/Remote
* Location - India.
* Fresher’s can apply.
* Good Communication Skills required.

Open Positions:
1.VlSI,IOT,AIML 
2.Full stack Web Developer 
3. Python Developer 
4. Associate Software Engineer
5. Junior Data Analyst
6. Management Trainee - Marketing
7.Java Developer 
8. IT Intern - Cybersecurity
9.UIUX Designer 
"""

# use sensible defaults for fields you didn't collect from UI
predict_job_posting(
    job_text,
    model=xgb_model,
    employment_type='Unknown',
    required_experience='Unknown',
    required_education='Unknown',
    telecommuting=1,         # set to 1 if the post advertises remote
    has_company_logo=0,
    has_questions=0
)
