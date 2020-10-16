import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from PIL import Image


def loadData():
    df = pd.read_csv("train.csv")
    return df

def preprocessing(df):
    X = df.iloc[:, [0,3,5]].values
    y = df.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y.flatten())

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
    return X_train, X_test, y_train, y_test, le

def randomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    score = accuracy_score(y_test, y_pred)*100
    report = classification_report(y_test, y_pred)
    return score, report, rf

def neuralNetwork(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2))
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred)
    return acc, rep, mlp
def decisionTree(X_train, X_test, y_train, y_test):

    tree = DecisionTreeClassifier(max_leaf_nodes=3)
    tree.fit(X_train,y_train)
    y_pred = tree.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return score, report, tree

def user_input():
    duration = st.text_input("Enter the Bike Ride Duration: ")
    start = st.text_input("Enter the start station: ")
    end = st.text_input("Enter the end station: ")
    data = np.array([duration, start, end])
    return data
def showMap():
    plotData = pd.read_csv("locations.csv")
    data = pd.DataFrame()
    data['lat'] = plotData['lat']
    data['lon'] = plotData['lon']
    return data

def main():
    st.title("Trip History Predictions by Vipul Gaurav")
    #image = Image.open('index.gif')
    #st.image(image, use_column_width=True)
    st.markdown("![Alt Text](https://raw.githubusercontent.com/vgaurav3011/Bike-Sharing-Demand-Analysis/master/assets/index.gif)")
    data = loadData()
    X_train, X_test, y_train, y_test, le = preprocessing(data)

    if st.checkbox("Display Data Frames"):
        st.subheader("Raw Data:")
        st.write(data.head())
    choose_model = st.sidebar.selectbox("Choose ML Model: ",
    ["Not Needed", "Decision Tree", "Random Forest", "NN"])

    if(choose_model=="Random Forest"):
        score, report, rf = randomForest(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Model: ")
        st.write(score, "%")
        st.text("Report of the Model is: ")
        st.write(report)

        try:
            if(st.checkbox("Enter your own data")):
                data = user_input()
                scaler = StandardScaler()
                scaler.fit(X_train)
                data = scaler.transform(data)
                pred = rf.predict(data)
                st.write("Class: ", le.inverse_transform(pred))
        except:
            pass
    elif(choose_model=="Neural Network"):
        score, report, clf = neuralNetwork(X_train, X_test, y_train, y_test)
        st.text("Accuracy of the model: ")
        st.write(score, "%")
        st.text("Report: ")
        st.write(report)

        try:
            if(st.checkbox("Enter your own data: ")):
                data = user_input()
                pred = clf.predict(data)
                st.write("Class: ", le.inverse_transform(pred))
        except:
            pass
    elif(choose_model=="Decision Tree"):
        score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
        st.text("Accuracy of the model: ")
        st.write(score, "%")
        st.text("Report: ")
        st.write(report)

        try:
            if(st.checkbox("Enter your own data: ")):
                data = user_input()
                pred = tree.predict(data)
                st.write("Class: ", le.inverse_transform(pred))
        except:
            pass
    # Mapping the data and predictions

    plotData = showMap()
    st.subheader("Bike Travel History: ")
    st.map(plotData, zoom=14)

    choose_vis = st.sidebar.selectbox("Choose Visualization: ",
    ["NONE", "Vehicles from Start", "Vehicles from End", "Count of Members"])

    if(choose_vis=="Vehicles from Start"):
        fig = px.histogram(data["Start station"], x="Start station")
        st.plotly_chart(fig)
    elif(choose_vis=="Vehicles from End"):
        fig = px.histogram(data['End station'], x='End station')
        st.plotly_chart(fig)
    elif(choose_vis=='Count of Members'):
        fig = px.histogram(data['Member type'], x='Member type')
        st.plotly_chart(fig)
    st.pyplot()
    
if __name__=='__main__':
    main()