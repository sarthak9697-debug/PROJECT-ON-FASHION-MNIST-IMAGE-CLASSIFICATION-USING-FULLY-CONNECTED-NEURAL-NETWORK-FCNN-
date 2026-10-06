import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------
# Fashion MNIST FCNN - Streamlit Deployment App
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="wide"
)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

MODEL_PATH = Path(__file__).parent / "model.pkl"


@st.cache_resource
def load_model():
    """
    Load model.pkl.

    The app supports both:
    1. A directly pickled Keras Sequential model.
    2. A dictionary bundle containing the model and class_names.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl was not found. Upload model.pkl in the same folder as app.py."
        )

    with open(MODEL_PATH, "rb") as model_file:
        saved_object = pickle.load(model_file)

    # Current model.pkl contains the Keras Sequential model directly.
    if hasattr(saved_object, "predict"):
        return saved_object, CLASS_NAMES

    # Also support the earlier dictionary-bundle format.
    if isinstance(saved_object, dict) and "model" in saved_object:
        model = saved_object["model"]
        class_names = saved_object.get("class_names", CLASS_NAMES)
        return model, class_names

    raise ValueError(
        "model.pkl contains an unsupported object. "
        "Expected a trained Keras model or a dictionary containing 'model'."
    )


def prepare_image(uploaded_image):
    """Convert uploaded image into the 28x28 normalized FCNN input."""
    image = Image.open(uploaded_image).convert("L")
    image = image.resize((28, 28))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = image_array / 255.0

    # Fashion MNIST FCNN expects one image with shape (1, 28, 28).
    image_array = image_array.reshape(1, 28, 28)

    return image, image_array


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("👕 Fashion MNIST Image Classifier")

st.write(
    "Upload a clothing image and the trained Fully Connected Neural Network "
    "(FCNN) will predict its Fashion MNIST category."
)

st.info(
    "Recommended input: a clear clothing item on a simple background. "
    "The image is converted to grayscale, resized to 28×28 pixels, "
    "and normalized from 0–255 to 0–1."
)

# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------

try:
    model, class_names = load_model()
except Exception as error:
    st.error(f"Unable to load model.pkl: {error}")
    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("📌 Model Information")
    st.write("**Model:** Fully Connected Neural Network (FCNN)")
    st.write("**Input:** 28 × 28 grayscale image")
    st.write("**Pixel range:** 0–255")
    st.write("**Model input range:** 0–1")
    st.write("**Output classes:** 10")

    st.markdown("---")
    st.subheader("Fashion MNIST Classes")

    for index, class_name in enumerate(class_names):
        st.write(f"{index} — {class_name}")


# ---------------------------------------------------------
# Image upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["png", "jpg", "jpeg"],
    help="Upload a PNG or JPG/JPEG image containing one clothing item."
)

if uploaded_file is not None:

    original_image, model_input = prepare_image(uploaded_file)

    left_column, right_column = st.columns([1, 2])

    with left_column:
        st.subheader("Uploaded Image")
        st.image(
            original_image,
            caption="Converted to 28×28 grayscale",
            width=280
        )

    with right_column:
        st.subheader("Prediction")

        if st.button("🔍 Predict Clothing Category", type="primary"):

            try:
                probabilities = np.asarray(
                    model.predict(model_input, verbose=0)
                )

                if probabilities.ndim != 2:
                    st.error(
                        "The model output has an unexpected shape. "
                        "Please verify that model.pkl is the trained Fashion MNIST FCNN model."
                    )
                    st.stop()

                probabilities = probabilities[0]

                if len(probabilities) != len(class_names):
                    st.error(
                        f"The model returned {len(probabilities)} outputs, "
                        f"but {len(class_names)} Fashion MNIST classes are expected."
                    )
                    st.stop()

                predicted_index = int(np.argmax(probabilities))
                predicted_class = class_names[predicted_index]
                confidence = float(probabilities[predicted_index])

                st.success(
                    f"Predicted Category: **{predicted_class}**"
                )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.subheader("Class Probabilities")

                probability_data = {
                    class_names[index]: float(probabilities[index])
                    for index in range(len(class_names))
                }

                st.bar_chart(probability_data)

                st.caption(
                    "Confidence represents the model's predicted probability "
                    "for the selected category."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# ---------------------------------------------------------
# Usage information
# ---------------------------------------------------------

with st.expander("📖 How to use this app"):

    st.markdown(
        """
1. Upload a clothing image in PNG or JPG/JPEG format.
2. The image is converted to grayscale.
3. The image is resized to **28×28 pixels**.
4. Pixel values are normalized from **0–255** to **0–1**.
5. The trained FCNN predicts one of the 10 Fashion MNIST categories.
6. The predicted category, confidence, and class probabilities are displayed.

**Example categories:** T-shirt/top, Trouser, Pullover, Dress, Coat,
Sandal, Shirt, Sneaker, Bag, and Ankle boot.
"""
    )

st.caption("Fashion MNIST FCNN deployment app")
