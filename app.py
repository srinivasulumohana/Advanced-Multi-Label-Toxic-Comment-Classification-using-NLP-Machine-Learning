import streamlit as st
import joblib
import os
from PIL import Image

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(

    page_title="Advanced Toxic Comment Classification",

    layout="wide"

)

# =========================================================
# TITLE
# =========================================================

st.title("🧠 Advanced Toxic Comment Classification")

st.write(
    "Multi-Label NLP Classification with Best Model Selection"
)

# =========================================================
# LOAD BEST MODEL
# =========================================================

MODEL_PATH = "models/best_toxic_model.pkl"

if not os.path.exists(MODEL_PATH):

    st.error(
        "Best model not found! Run advanced_train_model.py first."
    )

    st.stop()

model = joblib.load(MODEL_PATH)

# =========================================================
# LABELS
# =========================================================

labels = [

    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"

]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "Prediction",

        "Visualizations"

    ]

)

# =========================================================
# PREDICTION PAGE
# =========================================================

if page == "Prediction":

    st.header("Real-Time Toxic Comment Prediction")

    text_input = st.text_area(

        "Enter Comment",

        height=200

    )

    if st.button("Predict Labels"):

        if text_input.strip() == "":

            st.warning(
                "Please enter text"
            )

        else:

            prediction = model.predict(
                [text_input]
            )

            prediction = prediction[0]

            predicted_labels = []

            for i in range(len(labels)):

                if prediction[i] == 1:

                    predicted_labels.append(
                        labels[i]
                    )

            if len(predicted_labels) == 0:

                st.success(
                    "No toxic labels detected"
                )

            else:

                st.error(

                    f"Predicted Labels: {', '.join(predicted_labels)}"

                )

# =========================================================
# VISUALIZATION PAGE
# =========================================================

if page == "Visualizations":

    st.header("Advanced Data Visualizations")

    visualization_files = [

        "label_distribution.png",

        "heatmap.png",

        "text_length_distribution.png",

        "model_comparison.png"

    ]

    for file in visualization_files:

        path = os.path.join(
            "visualizations",
            file
        )

        if os.path.exists(path):

            image = Image.open(path)

            st.image(

                image,

                caption=file,

                use_container_width=True

            )

# =========================================================
# FOOTER
# =========================================================

st.write("---")

st.write(
    "Developed using Python, NLP, Scikit-learn & Streamlit"
)













































