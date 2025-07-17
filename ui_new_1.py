import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
from io import BytesIO

# Load models and scaler
def load_models_and_scaler():
    with open("level1_rf_model.pkl", "rb") as f1:
        level1_model = pickle.load(f1)
    with open("model.pkl", "rb") as f2:
        level2_model = pickle.load(f2)
    with open("scaler.pkl", "rb") as f3:
        scaler = pickle.load(f3)
    return level1_model, level2_model, scaler

# Interpret feature status
def interpret_feature(value, feature_name):
    thresholds = {
        "AMH": (1.5, 4.0),
        "HCG": (0.1, 5.0),
        "Waist": (28, 35),
        "Cycle": (21, 35),
        "Ratio": (0.75, 0.85),
    }
    low, high = thresholds[feature_name]
    if value < low:
        return "🔵 Low"
    elif value > high:
        return "🔴 High"
    else:
        return "🟢 Normal"

# PDF download function
def generate_download_link(result_str):
    b = BytesIO()
    b.write(result_str.encode())
    b.seek(0)
    b64 = base64.b64encode(b.read()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="pcos_result.txt">📄 Download Report</a>'
    return href

# Batch prediction function
def batch_predict(df):
    level1_model, level2_model, scaler = load_models_and_scaler()
    scaled_data = scaler.transform(df.values)
    level1_probs = level1_model.predict_proba(df.values)[:, 1] 
    final_input = np.hstack((scaled_data, level1_probs.reshape(-1, 1)))
    preds = level2_model.predict(final_input)
    probas = level2_model.predict_proba(final_input)[:, 1]
    df_result = df.copy()
    df_result['Prediction'] = preds
    df_result['Confidence'] = probas.round(2)
    return df_result

# Main UI
def main():
    st.set_page_config(page_title="PCOS Predictor", layout="wide")
    st.title("🧬 PCOS Prediction Web App")
    st.markdown("Enter your clinical data or upload a file for batch prediction.")

    tabs = st.tabs(["Single Prediction", "Batch Prediction", "About"])

    # --- Tab 1: Single Prediction ---
    with tabs[0]:
        with st.sidebar:
            st.header("📋 Input Features")
            amh = st.number_input("AMH (ng/mL)", min_value=0.0, format="%.2f")
            hcg = st.number_input("II beta-HCG (mIU/mL)", min_value=0.0, format="%.2f")
            waist = st.number_input("Waist (inch)", min_value=0.0, format="%.2f")
            cycle = st.number_input("Cycle length (days)", min_value=0.0, format="%.2f")
            ratio = st.number_input("Waist:Hip Ratio", min_value=0.0, format="%.2f")

        if st.button("🔍 Predict"):
            level1_model, level2_model, scaler = load_models_and_scaler()
            input_5 = np.array([[amh, hcg, waist, cycle, ratio]])
            scaled_5 = scaler.transform(input_5)
            level1_prob = level1_model.predict_proba(input_5)[0][1]
            input_final = np.append(scaled_5[0], level1_prob*0.3).reshape(1, -1)
            pred = level2_model.predict(input_final)[0]
            prob = level2_model.predict_proba(input_final)[0][1]

            st.subheader("🧾 Result")
            if 0.45 <= prob <= 0.55:
                st.warning("❔ Prediction is uncertain. Please consult a doctor.")
            elif pred == 1:
                st.error("⚠️ You have PCOS.")
            else:
                st.success("✅ You do not have PCOS.")
            st.write(f"**Prediction Confidence:** {prob:.2f}")

            st.subheader("📊 Feature Status Summary")
            st.write(f"AMH: {interpret_feature(amh, 'AMH')}")
            st.write(f"II beta-HCG: {interpret_feature(hcg, 'HCG')}")
            st.write(f"Waist: {interpret_feature(waist, 'Waist')}")
            st.write(f"Cycle Length: {interpret_feature(cycle, 'Cycle')}")
            st.write(f"Waist:Hip Ratio: {interpret_feature(ratio, 'Ratio')}")
            st.write(f"Level 1 RF Probability: {level1_prob:.2f}")

            result_text = f"""
            PCOS Prediction Result\n
            Input Values:\n
            AMH: {amh} ng/mL\nHCG: {hcg} mIU/mL\nWaist: {waist} inch\nCycle Length: {cycle} days\nWaist:Hip Ratio: {ratio}\n
            Final Prediction: {'PCOS Detected' if pred==1 else 'No PCOS'}\nConfidence: {prob:.2f}
            """
            st.markdown(generate_download_link(result_text), unsafe_allow_html=True)

    # --- Tab 2: Batch Prediction ---
    with tabs[1]:
        st.write("📂 Upload a CSV file with columns: AMH, HCG, Waist, Cycle, Ratio")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            try:
                result_df = batch_predict(df)
                st.dataframe(result_df)
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results CSV", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # --- Tab 3: About ---
    with tabs[2]:
        st.markdown("""
        ### 🧠 About This App
        This web app predicts the likelihood of Polycystic Ovary Syndrome (PCOS) using clinical features.

        **Model Pipeline**:
        - Level 1: Random Forest Classifier (Raw Input)
        - Level 2: Final Classifier (Combined Scaled Features + RF Probability)

        **Built with:** Streamlit | Scikit-learn | Python
        """)

if __name__ == "__main__":
    main()