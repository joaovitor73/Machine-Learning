import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas
import numpy as np
from joblib import load

st.subheader("Input")
result = st_mnist_canvas()

model_sgd = load('models/model-sgd.pkl')

if result.is_submitted:

    # Prepare the image for ML model prediction
    image_for_prediction = np.expand_dims(result.resized_grayscale_array, axis=0)

    image_for_prediction = image_for_prediction.reshape(1, -1)

    # Predict digit using a machine learning model
    prediction = model_sgd.predict(image_for_prediction)

    if  prediction[0]:
        st.write("O valor digitado é 5")
    else:
        st.write("O valor digitado não é 5")
    st.caption('IA pode cometer erros. Considere verificar informações importantes')