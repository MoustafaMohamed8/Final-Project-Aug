import streamlit as st
import joblib
import numpy as np
from utils import process_new

## Load the model

model=joblib.load('knn_model.pkl')


def Diabetes_classification():
    ## Title
    st.title('Diabetes Classification Perdection')
    st.markdown('---')

    ## input fields
    Pregnancies=st.number_input('How many Times did you get pregnant?',step=1)
    Glucose=st.number_input('What is your glucose level',step=1)
    BloodPressure=st.number_input('What is your BloodPressure?',step=1)
    SkinThickness=st.number_input('What is your SkinThickness reading?',step=1)
    Insulin=st.number_input('What is your Insulin Level',step=1)
    BMI=st.number_input('What is your Current BMI',step=1)
    DiabetesPedigreeFunction=st.number_input('Your Diabetes Pedigree Score',value=0.1,step=0.1)
    Age=st.number_input('How old are you?',value=14,step=1)
    BloodPressureCategory=st.selectbox('what does your BloodPressure Categorized in?',options=['Normal','Hypertension','Prehypertension'])
    BMICategory=st.selectbox('What Does your BMI Says about you? 18.5-Underweight , 24.9-Normal Weight ,  29.9-OverWeight ,\n Above is Obese',options=['Obese','Normal Weight','Underweight','OverWeight'])
    Log_Insulin=np.log1p(Insulin)
    Log_SkinThickness=np.log1p(SkinThickness)
    st.markdown('---')

    if st.button('Predict Whether you may be diabetic or Not'):
        new_data=np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,BloodPressureCategory,
                          BMICategory,Log_Insulin,Log_SkinThickness])

        X_processed=process_new(x_new=new_data)

    ## Predict
        y_pred=model.predict(X_processed)
        y_pred=bool(y_pred)

    ## Display
        st.success(f'Diabetes Prediction is {y_pred} ')
    
    return None



if __name__=='__main__':
    Diabetes_classification()