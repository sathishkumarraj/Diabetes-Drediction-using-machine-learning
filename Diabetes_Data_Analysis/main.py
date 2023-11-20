import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split

# Add a background image using custom CSS
st.set_page_config(
                   page_icon='icon',
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """# This OCR app is created by `SATHISH!`"""})

st.markdown( " " ,unsafe_allow_html=True)


# SETTING-UP BACKGROUND IMAGE
def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background:url("https://images.pexels.com/photos/305821/pexels-photo-305821.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
                        background-size: cover}}
                     </style>""" ,unsafe_allow_html=True)

setting_bg()


# The rest of your Streamlit app goes here
# st.title("My Streamlit App")

# loading the saved model
# loading the saved model
with open(r"D:\Data Science\Merit Skill intern\Datasets\Project 2 - Diabetes Data\Diabetes_Data_Analysis\trained_model.sav", 'rb') as file:
    loaded_model = pickle.load(file)


def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    numeric_data = np.array([float(x) for x in input_data])
    # input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = numeric_data.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'



def main():
    # giving a title
    st.title(':violet[Diabetes Prediction Web App]')

    # getting the input data from the user
    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.text_input(':blue[Number of Pregnancies]')
        if Pregnancies != '':
            try:
                numeric_value = float(Pregnancies)
                st.write(f"Pregnancies value entered: {numeric_value}")
            except ValueError:
                st.warning("Please enter a valid Pregnancies value.")



        Glucose = st.text_input(':blue[Glucose Level]')
        if Glucose != '':
            try:
                numeric_value = float(Glucose)
                st.write(f"Glucose value entered: {numeric_value}")
            except ValueError:
                st.warning("Please enter a valid Glucose value.")


        BloodPressure = st.text_input(':blue[Blood Pressure value]')
        if BloodPressure != '':
            try:
                integer_value = float(BloodPressure)
                st.write(f"BloodPressure value entered: {integer_value}")
            except ValueError:
                st.warning("Please enter a valid BloodPressure.")


        SkinThickness = st.text_input(':blue[Skin Thickness value]')
        if SkinThickness != '':
            try:
                integer_value = float(SkinThickness)
                st.write(f"SkinThickness value entered: {integer_value}")
            except ValueError:
                st.warning("Please enter a valid SkinThickness.")

    with col2:
        Insulin = st.text_input(':blue[Insulin Level]')
        if Insulin != '':
            try:
                integer_value = float(Insulin)
                st.write(f"Insulin value entered: {integer_value}")
            except ValueError:
                st.warning("Please enter a valid Insulin.")


        BMI = st.text_input(':blue[BMI value]')
        if BMI != '':
            try:
                integer_value = float(BMI)
                st.write(f"BMI value entered: {integer_value}")
            except ValueError:
                st.warning("Please enter a valid BMI.")

        DiabetesPedigreeFunction = st.text_input(':blue[Diabetes Pedigree Function value]')
        if DiabetesPedigreeFunction != '':
            try:
                integer_value = float(DiabetesPedigreeFunction)
                st.write(f"DiabetesPedigreeFunction value entered: {integer_value}")
            except ValueError:
                st.warning("Please enter a valid DiabetesPedigreeFunction.")

        Age = st.text_input(':blue[Age of the Person]')
        # Check if the input is not empty before converting to int
        if Age != '':
            try:
                integer_value = int(Age)
                st.write(f"Age value entered: {integer_value}")
            except ValueError:
                st.warning("Please enter a valid Age.")

        # Age1 = int(Age)

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()




