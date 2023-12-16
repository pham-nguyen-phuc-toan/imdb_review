import streamlit as st
from PIL import Image
import pickle as pkl
import tensorflow as tf
from tensorflow import keras
import numpy as np

class_list = {'0': 'Negative', '1': 'Positve'}

st.title('Sentiment analysis from IMDB review')

image = Image.open('feedback.jpg')
st.image(image)

input_md = open('bilstm_imdb.pkl', 'rb')
model = pkl.load(input_md)

st.header('Write a feedback')
txt = st.text_area('', '')

if txt != '':
    if st.button('Predict'):
        feature_vector = np.expand_dims([txt], axis=2)
        label = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(class_list[label])
