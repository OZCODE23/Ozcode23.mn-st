import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import base64
from streamlit_drawable_canvas import st_canvas

# MNIST modelini yükle
@st.cache_resource
def load_mnist_model():
    model = load_model("mnist_model.h5")  # Kaydedilmiş modeli yükle
    return model

model = load_mnist_model()

st.markdown("<h1 style='text-align: center; color: red; font-weight: bold;'>MNIST SAYI TAHMİNİ</h1>", unsafe_allow_html=True)

# Arka plan resmini ekleme
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
        }}
        .stMarkdown {{
            color: white;
            font-size: 20px;
            font-weight: bold;
        }}
        .button-css {{
            color: white !important;
            background-color: black !important;
            border-radius: 5px;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.jpg")  # Arka plan resmini burada belirtin

st.markdown("<h2 style='color: black; font-weight: bold;'>Lütfen bir resim yükleyin veya aşağıya çizim yapın.</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image)  # Resmi ters çevir
    image = image.resize((28, 28), Image.LANCZOS)  # Resmi 28x28 boyutuna getir

    st.image(image, caption='Yüklenen Resim (28x28)', width=150)

    image = np.array(image)
    image = image / 255.0  # Normalizasyon
    image = image.reshape(1, 28, 28, 1)  # Giriş verisini doğru şekilde dönüştür

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    st.markdown(f"<h2 style='color: white; font-weight: bold;'>Tahmin Edilen Rakam (Resim Yükleme): {predicted_class}</h2>", unsafe_allow_html=True)
else:
    st.write("Yüklenen resim yok.")

# Orta kısımda bir kolon oluşturuyoruz
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    # Çizim tuvali oluşturma
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Sadece siyah kullanarak çizim yapma
        stroke_width=20,  # Çizim kalınlığı
        stroke_color="#000000",  # Çizim rengi
        background_color="#FFFFFF",  # Arka plan rengi
        update_streamlit=True,
        height=560,
        width=560,
        drawing_mode="freedraw",
        key="canvas",
    )

# Orta kolonda Tahmin Et butonunu ekliyoruz
with col2:
    if canvas_result.image_data is not None:
        if st.button('Tahmin Et (Resim Çizme)', key="predict_button", help="button-css"):  # Button'a CSS class'ı ekledik
            # Canvas'tan çizilen görüntüyü al
            image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')

            # Resmi 28x28 boyutuna getir
            image = image.resize((28, 28), Image.LANCZOS)

            # Resmi siyah beyaz yap ve ters çevir
            image = ImageOps.invert(image.convert('L'))

            # Resmi güncelle ve yeniden normalize et
            image = np.array(image)
            image = image / 255.0  # Normalizasyon
            image = image.reshape(1, 28, 28, 1)  # Giriş verisini doğru şekilde dönüştür

            # Tahmini yap
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            st.markdown(f"<h2 style='color: white; font-weight: bold;'>Tahmin (Resim Çizme): {predicted_class}</h2>", unsafe_allow_html=True)

            # Görüntüyü yeniden göster (28x28 boyutunda)
            st.image(image.reshape(28, 28), caption='Rescaled Image (28x28)', width=150)