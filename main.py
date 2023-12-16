import streamlit as st
from PIL import Image
import pickle as pkl
import tensorflow as tf
from tensorflow import keras
from keras.datasets import imdb
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

maxlen = 200

index = imdb.get_word_index()

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
        sen = 'this film is so awful'
        sen = sen.lower()
        x_test = []
        for w in sen.split():
            x_test.append(index[w])
        pred = model.predict(feature_vector)
        label = np.argmax(pred, axis = -1)

        st.header('Result')
        st.text(class_list[label])
