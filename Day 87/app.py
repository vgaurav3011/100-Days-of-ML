import streamlit as st
import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import joblib

from PIL import Image

@st.cache
def load_dataset(dataset):
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(dataset, names=columns)
    return df

def load_prediction_models(model_file):
    load_model = joblib.load(open(os.path.join(model_file), "rb"))
    return load_model

buying_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
maint_label = {'vhigh': 0, 'low': 1, 'med': 2, 'high': 3}
doors_label = {'2': 0, '3': 1, '5more': 2, '4': 3}
persons_label = {'2': 0, '4': 1, 'more': 2}
lug_boot_label = {'small': 0, 'big': 1, 'med': 2}
safety_label = {'high': 0, 'med': 1, 'low': 2}
class_label = {'good': 0, 'acceptable': 1, 'very good': 2, 'unacceptable': 3}

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val==value:
            return key

def main():
    
    st.title("Car Value Evaluation using Machine Learning")
    st.subheader("Made with <3 by Vipul Gaurav")
    image = Image.open('assets/index.jpeg')
    st.image(image, use_column_width=True)
    tasks = ['EDA', 'Prediction']

    choices = st.sidebar.selectbox("Select Task: ", tasks)

    if choices=='EDA':
        st.subheader("Explore Data!!")
        data = load_dataset("data/car.data")
        st.dataframe(data.head(5))
        if st.checkbox("Show 5 point summary:"):
            st.write(data.describe())
        if st.checkbox("Simple Value Plots: "):
            st.write(sns.countplot(data['class']))
            st.pyplot()

        if st.checkbox("Pie Plots"):
            columns = data.columns.tolist()
            if st.button("Generate Pie"):
                st.write(data.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
                st.pyplot()

    if choices=='Prediction':            
            st.subheader("Lets start with ML")
            buying = st.selectbox("Select Buying Level: ", tuple(buying_label.keys()))
            maint = st.selectbox("Select Maintainence Level: ", tuple(maint_label.keys()))
            doors = st.selectbox("Select Number of Doors: ", tuple(doors_label.keys()))
            persons = st.number_input("Select Number of Persons: ", 2, 10)
            lug_boot = st.selectbox("Select Luggage Boot: ", tuple(lug_boot_label.keys()))
            safety = st.selectbox("Select Safety", tuple(safety_label.keys()))
            k_buying = get_value(buying, buying_label)
            k_maint = get_value(maint, maint_label)
            k_doors = get_value(doors, doors_label)
            k_lug_boot = get_value(lug_boot, lug_boot_label)
            k_safety = get_value(safety, safety_label)


            final_data = {
                "buying":buying,
                "maint":maint,
                "doors":doors,
                "persons":persons,
                "lug_boot":lug_boot,
                "safety":safety,
            }
            st.subheader("options selected")
            st.json(final_data)

            st.subheader("Data Encoding")

            sample_data = [k_buying, k_maint, k_doors, persons, k_lug_boot, k_safety]
            st.write(sample_data)

            prep_data = np.array(sample_data).reshape(1,-1)
            model_choices = st.selectbox("Model Type", ['Logistic Regression', 'Random Forest', 'MLP Classifier'])
            if st.button('Evaluate'):
                if model_choices=='Logistic Regression':
                    pred = load_prediction_models("models/logit_model.pkl")
                    y_pred = pred.predict(prep_data)
                    st.write(y_pred)
                if model_choices=='Random Forest':
                    pred = load_prediction_models("models/rf_model.pkl")
                    y_pred = pred.predict(prep_data)
                    st.write(y_pred)
                if model_choices=="MLP Classifier":
                    pred = load_prediction_models("models/nn_model.pkl")
                    y_pred = pred.predict(prep_data)
                    st.write(y_pred)
                final_result = get_key(y_pred, class_label)
                st.success(final_result)
if __name__=='__main__':
    main()