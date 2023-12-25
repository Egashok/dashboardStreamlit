import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/laba4.csv')

st.title("Датасет smoke detector")

st.header("Тепловая карта с корреляцией между основными признаками")

plt.figure(figsize=(12, 8))
selected_cols = ['Humidity[%]', 'Temperature[C]', 'eCO2[ppm]', 'Pressure[hPa]','TVOC[ppb]','PM1.0','PM2.5','NC0.5','NC1.0','NC2.5','CNT']
selected_df = df[selected_cols]
sns.heatmap(selected_df.corr(), annot=True, cmap='coolwarm')
plt.title('Тепловая карта с корреляцией')
st.pyplot(plt)

st.header("Гистограммы")

columns = ['Temperature[C]','Raw H2','Raw Ethanol']

for col in columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)[col], bins=100, kde=True)
    plt.title(f'Гистограмма для {col}')
    st.pyplot(plt)

st.header("Ящики с усами ")
outlier = df[columns]
Q1 = outlier.quantile(0.25)
Q3 = outlier.quantile(0.75)
IQR = Q3-Q1
data_filtered = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]


for col in columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data_filtered[col])
    plt.title(f'{col}')
    plt.xlabel('Значение')
    st.pyplot(plt)

st.header("Круговая диаграмма целевого признака")
plt.figure(figsize=(8, 8))
df['Fire Alarm'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Fire Alarm')
plt.ylabel('')
st.pyplot(plt)
