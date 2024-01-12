import streamlit as st
import pandas as pd
import pickle
import numpy as np


uploaded_file = st.file_uploader("Выберите файл датасета")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Загруженный датасет:", df)

    st.title("Датасет smoke detector")

st.title("Получить предсказания пожара.")

st.header("UTC")
utc = st.number_input("Число:", value=1654733331)

st.header("Temperature[C]:")
temperature = st.number_input("Число:", value=9)

st.header("Humidity[%]")
humidity = st.number_input("Число:", value=56.86)

st.header("TVOC[ppb]]:")
tvoc = st.number_input("Число:", value=11.0)

st.header("eCO2[ppm:")
co2 = st.number_input("Число:", value=400)

st.header("Raw H2:")
h2 = st.number_input("Число:", value=13347.0)

st.header("Raw Ethanol:")
ethanol = st.number_input("Число:", value=20160)

st.header("Pressure[hPa]]:")
pressure = st.number_input("Число:", value=939.575)

st.header("PM1.0:")
pm1 = st.number_input("Число:", value=1.78)

st.header("PM2.5:")
pm25 = st.number_input("Число:", value=1.85	)

st.header("NC0.5:")
nc05 = st.number_input("Число:", value=12.25)

st.header("NC1.0:")
nc1 = st.number_input("Число:", value=1.911)

st.header("NC2.5")
nc25 = st.number_input("Число:", value=0.015)

st.header("CNT::")
cnt = st.number_input("Число:", value=3178)

data = pd.DataFrame({'UTC': [utc],
                    'Temperature[C]': [temperature],
                    'Humidity[%]': [humidity],
                    'TVOC[ppb]': [tvoc],
                    'eCO2[ppm]': [co2],
                    'Raw H2': [h2],
                    'Raw Ethanol': [ethanol],
                    'Pressure[hPa]': [pressure],
                    'PM1.0': [pm1],
                    'PM2.5': [pm25],
                    'NC0.5': [nc05],
                    'NC1.0': [nc1],
                    'NC2.5': [nc25],
                    'CNT': [cnt],          
                    })


button_clicked = st.button("Предсказать")

if button_clicked:
    print('xz')
    with open('models/KNN.pkl', 'rb') as file:
        lr = pickle.load(file)
    with open('models/Bagging.pkl', 'rb') as file:
        bagging_model = pickle.load(file)
    with open('models/Gradien.pkl', 'rb') as file:
        gradient_model = pickle.load(file)
    with open('models/Kmeans.pkl', 'rb') as file:
         kmeans_model= pickle.load(file)
    from tensorflow.keras.models import load_model
    nn_model = load_model('models/nn.h5')

    with open('models/Stacking.pkl', 'rb') as file:
        stacking_model = pickle.load(file)

    st.header("KNN:")
    pred =[]
    knn_pred = lr.predict(data)[0]
    pred.append(int(knn_pred))
    st.write(f"{knn_pred}")


    st.header("bagging:")
    bagging_pred = bagging_model.predict(data)[0]
    pred.append(int(bagging_pred))
    st.write(f"{bagging_pred}")

    st.header("gradient:")
    gradient_pred = gradient_model.predict(data)[0]
    pred.append(int(gradient_pred))
    st.write(f"{gradient_pred}")

    st.header("Perceptron:")
    nn_pred = round(nn_model.predict(data)[0][0])
    pred.append(nn_pred)
    st.write(f"{nn_pred}")

    st.header("Stacking:")
    stacking_pred = stacking_model.predict(data)[0]
    pred.append(int(stacking_pred))
    st.write(f"{stacking_model.predict(data)[0]}")

