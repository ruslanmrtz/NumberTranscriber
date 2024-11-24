import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

from model import predict
from PIL import Image

st.title('Нарисуйте цифру')

col1, col2 = st.columns(2)

with col1:
    # Настройка холста
    picture = st_canvas(
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

# Обработка изображения
with col2:
    button = st.button('Отгадать цифру')
    if button:
        grayscale_image = np.dot(picture.image_data[..., :3], [0.2989, 0.5870, 0.1140]).round()  # Учитываем RGB-яркость
        image = Image.fromarray(grayscale_image).resize((28, 28))
        X = (255 - np.array(image)) * 3
        prediction = predict(X)
        st.metric('Ваша цифра', prediction)

