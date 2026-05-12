# 🕵️‍♂️ Fake Job Classification System

A **machine learning web app** that detects **fraudulent job postings** using NLP (TF-IDF) and **XGBoost**, powered by **FastAPI** and a responsive frontend (HTML, CSS, JS).

---

## 🚀 Features
- Predicts whether a job post is **Fake** or **Legit**
- Built with **FastAPI + XGBoost**
- Clean UI with real-time prediction
- Explainable AI (`/explain` endpoint → SHAP-like insights)
- Ready for **Render deployment**

---

## 🧠 Tech Stack
| Layer | Tools |
|-------|--------|
| **Language** | Python 3.11 |
| **Framework** | FastAPI |
| **ML / NLP** | XGBoost, Scikit-learn, TF-IDF, NLTK |
| **Frontend** | HTML, CSS, JS |
| **Deployment** | Render (Cloud) |

---

## 🧮 Dataset
**Source:** [Fake Job Posting Prediction – Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

- 17,880 job listings (13 features + target)
- `fraudulent` → **Target variable (0 = Real, 1 = Fake)**
- Highly imbalanced (~5% fake)

---

## 🧩 Model Pipeline
1. **Preprocessing:**  
   - Text cleaning, stopword removal, stemming  
   - Missing values → filled with `Unknown` / empty string  
2. **Feature Engineering:**  
   - TF-IDF for text (10,000 features)  
   - OneHot for categorical columns  
   - Binary numeric fields added  
3. **Training:**  
   - XGBoost (`scale_pos_weight` for imbalance)  
4. **Evaluation:**  
   - Accuracy: 95.3%  
   - ROC-AUC: 0.985  
   - Fake recall: 0.91

---

## 💻 Run Locally
```bash
# 1. Clone repo
git clone https://github.com/<your-username>/fake-job-classification-system.git
cd fake-job-classification-system

# 2. Create & activate venv
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run app
uvicorn app:app --reload

