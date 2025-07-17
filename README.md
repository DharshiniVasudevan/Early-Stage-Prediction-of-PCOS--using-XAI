# Early-Stage-Prediction-of-PCOS-using-XAI

This project was developed as part of our academic curriculum and focuses on the early detection of **Polycystic Ovary Syndrome (PCOS)** using machine learning and explainable AI (XAI). Our aim was not just to build accurate models, but also to ensure transparency in how predictions are made — especially for sensitive health-related decisions.

---

## 👩‍💻 Team Members

- **Dharshini** (GitHub: https://github.com/DharshiniVasudevan)
- **Chandralekha** (GitHub: https://github.com/BugganaLekha)
- **Rida** (GitHub: https://github.com/Ridaa-26)

---

## Project Description

This is an end-to-end machine learning pipeline for predicting PCOS based on clinical data. We've taken special care to make the predictions explainable using XAI methods, so both doctors and patients can understand why a certain prediction was made.

---

## Key Features

- Cleaned and preprocessed real-world data with techniques like scaling and SMOTE
- Multiple feature selection strategies to improve performance
- Trained various ML models and compared their performance
- Built-in Explainable AI with **LIME** and **SHAP** to show how the models make decisions
- Deployed a working UI using **Streamlit** for easy interaction and testing

---

## Technologies Used

### Data Preprocessing
- **Borderline SMOTE** for handling class imbalance  
- **StandardScaler** for feature scaling  
- Feature selection using:
  - TOPCA (Top Principal Component Analysis)
  - TOMIM (Top Mutual Information)
  - OSSM (Optimal Subset Selection)

### Model Training
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest (Single & Two-Level)  
- XGBoost  

### Explainable AI
- **LIME** (Local Interpretable Model-Agnostic Explanations)  
- **SHAP** (SHapley Additive exPlanations)  

### Interface & Deployment
- **Streamlit** – to create a clean and interactive front-end for predictions

---

## Note

This is a team project developed as part of our coursework. While the project is functional and well-tested, some modules might still be under active improvement.

---

## Acknowledgments

Special thanks to our faculty and mentors who guided us throughout the project.

---

## License

This repository is intended for educational use only.