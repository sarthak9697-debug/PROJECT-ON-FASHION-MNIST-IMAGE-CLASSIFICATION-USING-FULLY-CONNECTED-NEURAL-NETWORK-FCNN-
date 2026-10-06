import pickle
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = Path(__file__).with_name("model.pkl")

st.set_page_config(page_title="Fashion MNIST FCNN Classifier", page_icon="👕", layout="centered")

@st.cache_resource
def load_bundle():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

st.title("👕 Fashion MNIST Image Classifier")
st.caption("Fully Connected Neural Network (FCNN) • Streamlit deployment demo")

try:
    bundle = load_bundle()
    if bundle.get("requires_tensorflow") or bundle.get("model") is None:
        st.error("A trained TensorFlow model is required. Run the notebook training cells and regenerate model.pkl.")
        st.stop()
    model = bundle["model"]
    class_names = bundle["class_names"]
except FileNotFoundError:
    st.error("model.pkl was not found. Keep model.pkl in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Could not load model.pkl: {e}")
    st.stop()

st.markdown("""
### Project goal
Upload a clothing image and predict its Fashion MNIST category.

**Input:** grayscale clothing image  
**Model input:** 28 × 28 pixels, normalized to 0–1  
**Output:** one of 10 Fashion MNIST classes
""")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["png", "jpg", "jpeg"],
    help="For best results, use a simple clothing image similar to Fashion MNIST."
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.subheader("Uploaded image")
    st.image(image, caption="Input image", width=280)

    if st.button("Classify Image", type="primary", use_container_width=True):
        processed = image.resize((28, 28))
        array = np.asarray(processed).astype("float32") / 255.0
        probabilities = model.predict(array.reshape(1, 28, 28), verbose=0)[0]

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        st.success(f"Prediction: {class_names[index]}")
        st.metric("Confidence", f"{confidence:.2%}")

        if confidence < 0.60:
            st.warning("Low confidence. Try a clearer, centered clothing image.")
        elif confidence >= 0.85:
            st.info("High-confidence prediction.")

        st.subheader("Class probabilities")
        st.dataframe(
            {class_names[i]: [f"{float(probabilities[i]):.2%}"] for i in range(10)},
            use_container_width=True
        )

st.divider()

with st.expander("Input guidance"):
    st.markdown("""
- Recommended: simple clothing image.
- Any reasonable image size is accepted; it is resized to **28 × 28**.
- RGB/RGBA images are converted to grayscale.
- Confidence **<60%**: interpret cautiously.
- Confidence **≥85%**: stronger prediction.
- Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
""")

with st.expander("About the model"):
    st.write(bundle.get("description", "Fashion MNIST FCNN classifier."))
    st.write("Architecture: Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax).")

with st.expander("Run locally"):
    st.code("pip install streamlit tensorflow pillow numpy\nstreamlit run app.py", language="bash")
    st.code("app.py\nmodel.pkl")

st.caption("Portfolio project • Explainable FCNN • Deployment-ready")
