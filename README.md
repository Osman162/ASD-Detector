# 🧠 ASD Detector — Autism Spectrum Disorder Detection Tool

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Language-Arabic-blue?style=flat-square"/>
</p>

> An ML-powered Arabic-language tool for early behavioral screening of Autism Spectrum Disorder (ASD) in children. Built as a Streamlit web app and integrated into a mobile application by a development team.

---

## 📸 Screenshots

> *(Mobile app integration screenshots coming soon)*

---

## ✨ Features

- 🤖 **Ensemble of 3 ML models** for robust prediction
- 📋 **10 behavioral questions** based on clinical ASD indicators
- 📊 **Detailed section analysis**: Social Play · Physical Activity · Communication
- 📥 **Downloadable report** with full breakdown
- 🌐 **Full Arabic interface** with RTL support
- ⚕️ **Medical disclaimer** built-in
- 📱 **Mobile app integration** (Flutter)

---

## 🤖 Model Architecture

| Model | Weight | Purpose |
|-------|--------|---------|
| Random Forest 🌲 | 90% | Primary predictor |
| Logistic Regression 📈 | 5% | Linear boundary |
| Naive Bayes 📊 | 5% | Probabilistic baseline |

The final probability is a **weighted ensemble**:
```
final_prob = 0.90 × RF + 0.05 × LR + 0.05 × NB
```

---

## 📂 Project Structure

```
asd-detector/
│
├── autism_detector_app.py      # Main Streamlit app
├── autism_detector_model.pkl   # Random Forest model
├── logistic_model.pkl          # Logistic Regression model
├── naive_bayes_model.pkl       # Naive Bayes model
├── cleaned_data.csv            # Training dataset
└── requirements.txt            # Dependencies
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run autism_detector_app.py
```

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
joblib
```

---

## 📊 Input Features

The model uses **13 features**:

| Category | Features |
|----------|---------|
| Behavioral (10) | Responses to ASD screening questions (Yes/No) |
| Demographic (3) | Age, Gender, Ethnicity |
| Additional (3) | Country, Relationship, App usage history |

---

## ⚠️ Disclaimer

This tool is intended for **preliminary screening purposes only** and does **not** replace professional medical diagnosis. Always consult a qualified specialist for accurate diagnosis.

---

## 👨‍💻 Author

**Ahmed Osman**
- 🔗 [LinkedIn](https://www.linkedin.com/in/ahmed-osman-11892a351/)
- 🐙 [GitHub](https://github.com/Osman162)

---

## 📄 License

MIT License — feel free to use, modify, and distribute with attribution.
