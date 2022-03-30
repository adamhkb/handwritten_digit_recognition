import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import time

def np_to_df(outputs):
    length = outputs.shape[0]
    arr = []
    for pos in range(0, length):
        line = [0]*10
        line[pos] = outputs[pos]
        arr.append(line)
    return arr

model = load_model('mnist2.h5')

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon=":pencil:",
)

st.title('Handwritten Digit Recognition')
st.markdown('''
Try to write a numerical digit!
''')


with st.sidebar:
    stroke_width_adjust = st.slider("Stroke width: ", 1, 50, 20)
    st.markdown("---")
    st.markdown(
        """
            <h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://adamhkb.github.io/portfolio/">Adam</a></h6>
            <br>
            <a href="https://github.com/adamhkb" target='_blank'><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Logo.png" alt="Streamlit logo" height="20"></a>
            <a href="https://www.linkedin.com/in/adamhkamarulbahrin/" target='_blank' style='margin-left: 10px;'><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/LinkedIn_Logo.svg/1000px-LinkedIn_Logo.svg.png" alt="Streamlit logo" height="26"></a>
            """,
        unsafe_allow_html=True,
    )



SIZE = 280
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=stroke_width_adjust,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))

if st.button('Predict!'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_x = test_x.reshape(1, 28, 28, 1)
    test_x = (test_x - np.mean(test_x))/np.std(test_x)
    outputs = model.predict(test_x).squeeze() ** 0.2
    ind_max = np.where(outputs == max(outputs))[0][0]
    chart_data = pd.DataFrame(np_to_df(outputs),
                              index=['0', '1', '2', '3', '4','5', '6', '7', '8', '9'],
                              columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
    st.markdown(f"<h3 style = 'text-align: center;'>Prediction : {ind_max}<h3>", unsafe_allow_html=True)
    st.bar_chart(chart_data)

    st.markdown("---")

    st.subheader(
        f'Model: Keras | Test-Accuracy: 99.64%')
    st.write(
        'Batchsize: 32, Epochs: 100')
    st.code(
        '''
model = Sequential()

model.add(Conv2D(filters=32, 
                 kernel_size=5, 
                 padding='same', 
                 input_shape=(28,28,1), 
                 activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, 
                 kernel_size=5,  
                 padding='same', 
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, 
                 kernel_size=3,  
                 padding='same',  
                 activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, 
                 kernel_size=3,  
                 padding='same', 
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
        '''
    )
