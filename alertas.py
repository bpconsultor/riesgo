import streamlit as st
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

with open('desercion_modelo.pickle', 'rb') as f:
    classifier = pickle.load(f)

with open('scalar.pickle', 'rb') as t:
    scaler = pickle.load(t)
def main():
    asistencia = st.sidebar.number_input('Asistencia')
    indice = st.sidebar.number_input('Indice Acumulado')
    edad = st.sidebar.number_input('Edad')
    lista = [[asistencia, indice, edad]]
    scaled_result = scaler.transform(lista)

    st.title('Predicción del riesgo de abandono')
    if st.button('Predecir'):
        abandono = classifier.predict(scaled_result)
        st.subheader(abandono)
        if abandono == 1:
            st.subheader('En Riesgo')
        else:
            st.subheader('No riesgo')


if __name__== '__main__':
    main()


