import streamlit as st
from streamlit_mnist_canvas import st_mnist_canvas
import numpy as np
import pickle

st.title("Reconhecimento de Dígitos 🧠")
st.subheader("Desenhe um número abaixo:")

# Canvas para desenhar o dígito
result = st_mnist_canvas()

# Carregar modelo + PCA
with open('models/best_model_KNN_PCA.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    pca = data['pca']

if result.is_submitted:
    # Converte imagem 28x28 para vetor (1x784)
    image_for_prediction = np.expand_dims(result.resized_grayscale_array, axis=0)
    image_for_prediction = image_for_prediction.reshape(1, -1)

    # Aplica o PCA treinado (mesmo usado no treino)
    image_for_prediction_pca = pca.transform(image_for_prediction)

    # Faz predição
    prediction = model.predict(image_for_prediction_pca)

    # Mostra o resultado
    st.subheader("Predição")
    st.write(f"O número desenhado é: {int(prediction[0])}")
    st.caption("⚠️ A IA pode errar — verifique o resultado.")
