# app.py

from enum import auto
import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image
import streamlit.components.v1 as components

# --- Page Setup and Pipeline Loading ---
st.set_page_config(page_title="Cloud Classification", page_icon="☁️", layout="wide")
st.title("Cloud Classification AI ☁️")

# --- Data Mappings ---
CLASS_NAMES = ['Ac', 'As', 'Cb', 'Cc', 'Ci', 'Cs', 'Ct', 'Cu', 'Ns', 'Sc', 'St']
CLOUD_INFO = {
    'Ac': ('Altocumulus', 'Böljemoln'), 'As': ('Altostratus', 'Skiktmoln'),
    'Cb': ('Cumulonimbus', 'Bymoln / Åskmoln'), 'Cc': ('Cirrocumulus', 'Makrillmoln'),
    'Ci': ('Cirrus', 'Fjädermoln'), 'Cs': ('Cirrostratus', 'Slöjmoln'),
    'Ct': ('Contrail', 'Kondensstrimma (från flygplan)'), 'Cu': ('Cumulus', 'Stackmoln'),
    'Ns': ('Nimbostratus', 'Regnmoln'), 'Sc': ('Stratocumulus', 'Valkmoln'),
    'St': ('Stratus', 'Dimmoln')
}
WIKIPEDIA_ARTICLES = {
    'Ac': 'Altocumulus', 'As': 'Altostratus', 'Cb': 'Cumulonimbus',
    'Cc': 'Cirrocumulus', 'Ci': 'Cirrusmoln', 'Cs': 'Cirrostratus',
    'Ct': 'Kondensationsstrimma', 'Cu': 'Cumulus', 'Ns': 'Nimbostratus',
    'Sc': 'Stratocumulus', 'St': 'Stratus_(moln)'
}

# --- Load the saved Pipeline file ---
@st.cache_resource
def load_pipeline(pipeline_path):
    """Loads the entire saved Scikit-learn pipeline."""
    try:
        pipeline = joblib.load(pipeline_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"ERROR: Model file '{pipeline_path}' could not be found. Make sure it is in the same folder as app.py.")
        return None

# Update the filename to the exact one from the notebook
pipeline = load_pipeline('')


# --- Advanced Preprocessing Function ---
def preprocess_image_for_pipeline(image_pil):
    """
    Prepares an uploaded image in the exact same way as in the training notebook.
    This includes creating a 4-channel image (RGB + Gray).
    """
    # Convert from PIL Image (RGB) to an OpenCV-readable NumPy array
    image_rgb = np.array(image_pil)
    
    # 1. Create a grayscale version
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Reshape the grayscale image to have a 3rd dimension for concatenation
    gray_image_reshaped = gray_image.reshape(gray_image.shape + (1,))
    
    # 3. Concatenate the original RGB image (3 channels) with the grayscale image (1 channel)
    # The result is a 4-channel image.
    four_channel_image = np.concatenate((image_rgb, gray_image_reshaped), axis=2)

    # 4. Resize to the same size used during training (e.g., 128x128)
    img_size = 128 
    resized_image = cv2.resize(four_channel_image, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    
    # 5. Flatten the image into a single long 1D vector
    flattened_image = resized_image.flatten()
    
    # Return as a 2D array with one row, ready for the pipeline
    return flattened_image.reshape(1, -1)


# --- Application Interface ---
st.write("Ladda upp en bild på ett moln så försöker jag gissa vilken typ det är.")
uploaded_file = st.file_uploader("Välj en bildfil...", type=["jpg", "jpeg", "png", "webp"])

# Only run the app if the pipeline was loaded successfully
if pipeline is not None and uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    with st.spinner('Analyserar bilden...'):
        # Use the advanced preprocessing function
        features = preprocess_image_for_pipeline(image)
        
        # Use the entire pipeline to make a prediction
        prediction_array = pipeline.predict(features)   # This returns e.g., ['Cu']
        predicted_abbreviation = prediction_array[0]     # This becomes the string 'Cu'
        full_name, swedish_name = CLOUD_INFO[predicted_abbreviation]
        article_name = WIKIPEDIA_ARTICLES[predicted_abbreviation]
        wiki_url = f"https://sv.wikipedia.org/wiki/{article_name}"

    # Create the layout 
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Din uppladdade bild")
        st.image(image, caption="Din uppladdade bild", use_container_width=True)
        st.success(f"Jag tror att detta är ett **{full_name} ({predicted_abbreviation})**-moln.")
        st.info(f"På svenska kallas denna molntyp för **{swedish_name}**.")

    with col2:
        st.subheader(f"Läs mer om {full_name} på Wikipedia")
        components.iframe(wiki_url, height=auto, scrolling=True)