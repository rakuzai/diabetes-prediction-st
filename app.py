import streamlit as st
import pickle
import numpy as np

# Load the model
with open('diabetes.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Function to make predictions
def predict_diabetes(input_data):
    prediction = loaded_model.predict(input_data)
    return prediction[0]  # Assuming the prediction is a single value

# Streamlit app
def main():
    st.title("Diabetes Prediction App with GaussianNB Algorithm")

    # Input features arranged in two columns
    col1, col2 = st.columns(2)

    # Input features in the first column
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        glucose = st.number_input("Glucose", 0, 200, 0)
        blood_pressure = st.number_input("Blood Pressure", 0, 150, 0)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 0)

    # Input features in the second column
    with col2:
        insulin = st.number_input("Insulin", 0, 300, 0)
        bmi = st.number_input("BMI", 0.0, 50.0, 0.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.0, 0.0)
        age = st.number_input("Age", 0, 100, 0)

    # Button column
    button_col1, button_col2 = st.columns(2)

    # Submit button
    with button_col1:
        if st.button("Submit"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            prediction = predict_diabetes(input_data)
            result_text = "Positive" if prediction == 1 else "Negative"
            st.write("Prediction:", result_text)

    # Reset button
    with button_col2:
        if st.button("Reset"):
            st.experimental_rerun()

if __name__ == '__main__':
    main()
