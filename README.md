# 🏏 IPL Match Outcome Predictor

An end-to-end ML pipeline that predicts IPL match outcomes using Random Forest.

## 📌 Features
- Predicts win probability for both teams
- Based on team, venue, toss winner and toss decision
- Trained on IPL data from 2008–2020

## 🛠️ Tech Stack
- Python, Pandas, Scikit-learn
- Streamlit (frontend + deployment)

## 📁 Project Structure
- `src/preprocess.py` — data cleaning & feature engineering
- `src/train.py` — model training & comparison
- `src/predict.py` — prediction logic
- `app.py` — Streamlit web app
