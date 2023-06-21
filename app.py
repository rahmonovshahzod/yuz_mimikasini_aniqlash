import streamlit as st
from fastai.vision.all import *
import pathlib 
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title("yuz mimikasini aniqlovchi dastur")

st.markdown('''
    Bu sizni qo'rqqan, jahli chiqqan, yoki xursand ekanligizni aniqlab beradi.
    Ishlatish uchun rasm yuklang va modelning natijalarini ko'ring.
''')

file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif'])

if file:
    img = PILImage.create(file)
    st.image(img)

    model = load_learner("mimika_model.pkl")

    pred, pred_idx, probs = model.predict(img)
    pred_class = model.dls.vocab[pred_idx]
    st.success(f"Bashorat: {pred_class}")
    st.info(f"aniqlik: {probs[pred_idx]}")

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
