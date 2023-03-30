import streamlit as st
from class_predict import MlPredict

# app
st.title('Heart Disease Prediction using ML')
st.subheader('Enter the below form to know if you have a heart disease: ')

with st.sidebar:
    st.subheader('Chose your ML model: ')
    choices = st.multiselect('Select your model', ['Random Forest Model (100%)', 'SVM Model (96%)'])

with st.form(key='form1'):
    age = st.slider('Pick your Age : ', 0, 130)
    gender = st.selectbox('Pick your Gender : ', ['Male', 'Female'])
    chest_pain = st.selectbox("Chest Pain type : ", [0, 1, 2, 3])
    trest_bps = st.slider('Resting Blood Pressure : ', 94, 200)
    cholestrol = st.slider('Cholestrol : ', 120, 600)
    fbs = st.selectbox('Fasting Blood sugar > 120mg/dl : ', ['Yes', 'no'])
    restecg = st.selectbox('Resting Electrocardiographic results : ', [0, 1, 2])
    thalach = st.slider('Maximum Heart rate achieved : ', 70, 250)
    exang = st.selectbox('Do you excersice including agina', ['Yes', 'No'])
    old_peak = st.text_input('What is your ST depression induced by exercise relative to rest')
    slope = st.selectbox('Slope of the peak exercise ST segment : ', [0, 1, 2])
    ca = st.selectbox('Ca : ', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thal: ', [0, 1, 2, 3])

    submit_button = st.form_submit_button(label='Predict')

    gender = 1 if gender == 'Male' else 0
    fbs = 1 if fbs == 'Yes' else 0
    exang = 1 if exang == 'Yes' else 0
    if old_peak:
        old_peak = float(old_peak)
    

    data = [age, gender, chest_pain, trest_bps, cholestrol, fbs, restecg, thalach, exang, old_peak, slope, ca, thal]

    if submit_button:          
        print(choices)
        model = MlPredict(data=data)
        predictions = model.predict(model_type=choices)
        if len(choices) == 2:
            if predictions[0] == 0:
                st.balloons()
                st.success(f'{choices[0]}: Congratulations,  You don\'t have Heart Disease, Stay Healthy!!!!')
            else:
                st.text(f'{choices[0]}: You have Heart Disease, please consult a doctor!!!')
            if predictions[1] == 0:
                st.balloons()
                st.success(f'{choices[0] if len(choices) != 0 else "Random Forest (100%)"}: Congratulations,  You don\'t have Heart Disease, Stay Healthy!!!!')
            else:
                st.text(f'{choices[0] if len(choices) != 0 else "Random Forest (100%)"}: You have Heart Disease, please consult a doctor!!!!')
        else:
            if predictions[0] == 0:
                st.balloons()
                st.success(f'{choices[0] if len(choices) != 0 else "Random Forest (100%)"}: Congratulations,  You don\'t have Heart Disease, Stay Healthy!!!!')       
            else:
                st.text(f'{choices[0] if len(choices) != 0 else "Random Forest (100%)"}, You have Heart Disease, please consult a doctor!!!')     