from enum import auto
import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image
import streamlit.components.v1 as components

# --- Sidans inställningar och modell-laddning ---
st.set_page_config(page_title="Molnklassificering", page_icon="☁️", layout="wide")
st.title("Molnklassificering AI ☁️")

# Mappningar och modell-laddning
CLASS_NAMES = ['Ac', 'As', 'Cb', 'Cc', 'Ci', 'Cs', 'Ct', 'Cu', 'Ns', 'Sc', 'St']
CLOUD_INFO = {
    'Ac': ('Altocumulus', 'Böljemoln'), 'As': ('Altostratus', 'Skiktmoln'),
    'Cb': ('Cumulonimbus', 'Bymoln / Åskmoln'), 'Cc': ('Cirrocumulus', 'Makrillmoln'),
    'Ci': ('Cirrus', 'Fjädermoln'), 'Cs': ('Cirrostratus', 'Slöjmoln'),
    'Ct': ('Contrail', 'Kondensstrimma (från flygplan)'), 'Cu': ('Cumulus', 'Stackmoln'),
    'Ns': ('Nimbostratus', 'Regnmoln'), 'Sc': ('Stratocumulus', 'Valkmoln'),
    'St': ('Stratus', 'Dimmoln')
}

# Wikipedia-artiklar på svenska för varje molntyp
WIKIPEDIA_ARTICLES = {
    'Ac': 'Altocumulus', 'As': 'Altostratus', 'Cb': 'Cumulonimbus',
    'Cc': 'Cirrocumulus', 'Ci': 'Cirrusmoln', 'Cs': 'Cirrostratus',
    'Ct': 'kondensationsstrimma', 'Cu': 'Cumulusmoln', 'Ns': 'Nimbostratus',
    'Sc': 'Stratocumulus', 'St': 'Stratusmoln'
}

# Ladda modellen med caching för att undvika omladdning vid varje interaktion
@st.cache_resource
def load_model(model_path):
    model = joblib.load(model_path)
    return model

model = load_model('cloud_classifier_model.joblib')

# --- Funktion för att förbereda en bild ---
def preprocess_image(image_pil):
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img_size = 128
    resized_image = cv2.resize(gray_image, (img_size, img_size))
    flattened_image = resized_image.flatten()
    return flattened_image

# --- Applikationens gränssnitt ---
st.write("Ladda upp en bild på ett moln så försöker jag gissa vilken typ det är.")
uploaded_file = st.file_uploader("Välj en bildfil...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Skapa två kolumner
    col1, col2 = st.columns([1, 2])

    # Vänster kolumn (bild + prediktion)
    with col1:
        st.subheader("Din uppladdade bild")
        image = Image.open(uploaded_file)
        st.image(image, caption="Din uppladdade bild", width='stretch')

        st.write("")
        with st.spinner('Tänker efter...'):
            features = preprocess_image(image)
            features_reshaped = features.reshape(1, -1)
            prediction_index = model.predict(features_reshaped)

            predicted_cloud = CLASS_NAMES[prediction_index[0]]
            full_name, swedish_name = CLOUD_INFO[predicted_cloud]

        st.success(f"Jag tror att detta är ett **{full_name} ({predicted_cloud})**-moln.")
        st.info(f"På svenska kallas denna molntyp för **{swedish_name}**.")

    # Höger kolumn (Wikipedia-artikel)
    with col2:
        st.subheader(f"Läs mer om {full_name} på Wikipedia")
        article_name = WIKIPEDIA_ARTICLES[predicted_cloud]
        wiki_url = f"https://sv.wikipedia.org/wiki/{article_name}"

        components.iframe(wiki_url, height=auto, scrolling=True)
