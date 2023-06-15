import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import joblib


def load_model_and_scaler():
    # Load the trained model and scaler from the saved pkl files
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler


def user_input():
    age = st.sidebar.slider('Age', 0, 80, 30)
    hypertension = st.sidebar.checkbox('Hypertension')
    heart_disease = st.sidebar.checkbox('Heart Disease')
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    HbA1c_level = st.sidebar.slider('HbA1c Level', 0.0, 14.0, 5.0)
    blood_glucose_level = st.sidebar.slider(
        'Blood Glucose Level', 70, 200, 100)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    smoking_history = st.sidebar.selectbox(
        'Smoking History', ['Current smoker', 'Former smoker', 'Never smoked', 'Unknown'])
    return {'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease, 'bmi': bmi,
            'HbA1c_level': HbA1c_level, 'blood_glucose_level': blood_glucose_level, 'gender': gender,
            'smoking_history': smoking_history}


def preprocess(user_input_dict):
    df = pd.DataFrame(user_input_dict, index=[0])
    df['age_group'] = pd.cut(df['age'], bins=[0, 12, 35, 50, 80])
    df['bmi_cat'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100])
    df.drop_duplicates(keep='first', inplace=True)
    df['gender'] = df['gender'].replace('Other', 'Female')
    df['smoking_history'] = df['smoking_history'].replace(
        ['not current', 'former'], ['Former smoker', 'Former smoker'])
    df['smoking_history'] = df['smoking_history'].replace(
        ['current', 'ever'], ['Current smoker', 'Current smoker'])
    df['smoking_history'] = df['smoking_history'].replace('No Info', 'Unknown')
    df['smoking_history'] = df['smoking_history'].replace(
        'never', 'Never smoked')
    df['age'] = df['age'].astype(int)
    df['hypertension'] = df['hypertension'].astype(bool)
    df['heart_disease'] = df['heart_disease'].astype(bool)
    df = pd.get_dummies(
        df, columns=['age_group', 'bmi_cat', 'gender', 'smoking_history'])

    # Define the desired order of columns
    desired_order = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
                     'blood_glucose_level', 'age_group_(0, 12]',
                     'age_group_(12, 35]', 'age_group_(35, 50]', 'age_group_(50, 80]',
                     'bmi_cat_(0.0, 18.5]', 'bmi_cat_(18.5, 24.9]', 'bmi_cat_(24.9, 29.9]',
                     'bmi_cat_(29.9, 100.0]', 'gender_Female', 'gender_Male',
                     'smoking_history_Current smoker', 'smoking_history_Former smoker',
                     'smoking_history_Never smoked', 'smoking_history_Unknown']

    # Reorder the DataFrame columns
    df = df.reindex(columns=desired_order, fill_value=0)

    # Remove the redundant features
    df = df.drop(['bmi_cat_(0.0, 18.5]', 'bmi_cat_(18.5, 24.9]',
                  'bmi_cat_(24.9, 29.9]', 'bmi_cat_(29.9, 100.0]',
                  'age_group_(0, 12]', 'age_group_(12, 35]',
                  'age_group_(35, 50]', 'age_group_(50, 80]'], axis=1)

    return df


def predict(model, scaler, input_df):
    X = scaler.transform(input_df.values)
    prediction = model.predict(X)
    return prediction


def main():
    st.title('Diabetes Prediction App')
    st.write(
        'Please provide your health information in the sidebar to get a prediction.')
    user_input_dict = user_input()
    model, scaler = load_model_and_scaler()
    input_df = preprocess(user_input_dict)
    prediction = predict(model, scaler, input_df)
    if prediction[0] == 1:
        st.write(
            'You may be at risk of developing diabetes. Please consult a doctor for medical advice.')
    else:
        st.write(
            'You seem to be healthy and not at risk of developing diabetes. Please maintain a healthy lifestyle.')


if __name__ == '__main__':
    main()
