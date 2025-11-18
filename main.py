import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("wheat_disease.h5")

model = load_model()

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((256, 256))   # (WIDTH, HEIGHT)
    img = np.array(img) / 255.0
    print("DEBUG SHAPE:", img.shape)  # <-- Print to verify
    img = np.expand_dims(img, axis=0)
    img = img.astype("float32")
    return img


st.title("🌾 Wheat Disease Detection")

uploaded_file = st.file_uploader("Upload a wheat leaf image", type=["jpg", "jpeg", "png"])

class_names = ["healthy", "septoria", "stripe_rust"]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    img = preprocess_image(image)

    preds = model.predict(img)
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    st.subheader("Prediction Result")
    st.write(f"**Predicted class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.subheader("Class Probabilities")
    for cls, prob in zip(class_names, preds[0]):
        st.write(f"{cls}: {prob:.3f}")
