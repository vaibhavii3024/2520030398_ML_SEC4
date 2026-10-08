import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="COVID-19 X-Ray Detection",
    page_icon="🫁",
    layout="centered"
)

# Title
st.title("🫁 COVID-19 X-Ray Detection")
st.write("VGG16 Based Chest X-Ray Classification System")

st.divider()

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid_vgg16_model.keras")

# Classes
classes = [
    "COVID",
    "Lung Opacity",
    "Normal",
    "Viral Pneumonia"
]

# Upload image
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        width=400
    )

    # Predict button
    if st.button("🔍 Predict"):

        # Load model
        model = load_model()

        # Resize image
        image_resized = image.resize((224, 224))

        # Convert image to array
        img_array = np.array(
            image_resized,
            dtype=np.float32
        )

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )

        # Get result
        predicted_index = np.argmax(
            prediction[0]
        )

        predicted_class = classes[
            predicted_index
        ]

        confidence = (
            prediction[0][predicted_index] * 100
        )

        # Display result
        st.success(
            f"Prediction: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # Show probabilities
        st.subheader("📊 Class Probabilities")

        for i, class_name in enumerate(classes):

            probability = (
                prediction[0][i] * 100
            )

            st.write(
                f"{class_name}: "
                f"{probability:.2f}%"
            )

            st.progress(
                float(prediction[0][i])
            )

st.divider()

st.caption(
    "Academic project using VGG16. "
    "This prediction is not a medical diagnosis."
)