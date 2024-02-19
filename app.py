# Import necessary libraries
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'C:\\Users\\Smit\\Desktop\\SFDS\\model.h5'  # Update with the path to your trained model file
model = load_model(model_path)

# Import necessary libraries
import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Set page title
st.title("Signature Forgery Detection")

# Set background color and left-align contents
st.markdown(
    """
    <style>
        body {
            background-color: #ffffff;  Ensure high specificity 
            margin: 0;  Remove default margin 
        }
        .element-container {
            max-width: none;  Remove inline style limitations 
            padding: 0;  Remove default padding 
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display information about CNN and steps
st.markdown(
    """
    Follow the following steps to identify signature authenticity:

    <div style="margin-left: 20px; margin-top: 20px;">
        Step 1: Select an image which you want to check
    </div>

    <div style="margin-left: 20px;">
        Step 2: Click on Verify and wait for results
    </div>
    """,
    unsafe_allow_html=True
)

# File uploader for image input
uploaded_file = st.file_uploader("Click here to Upload Signature (only .jpg format)", type=["jpg"])
st.markdown("<style>div[data-baseweb='tooltip-container']{margin-top: 20px;}</style>", unsafe_allow_html=True)

# Verify button to save the image in "verification" folder
if st.button("Verify"):
    if uploaded_file is not None:
        # Specify the folder to save the image (outside the current folder)
        verification_folder = "C:\\Users\\Smit\\Desktop\\SFDS\\Customized_Dataset\\verification"
        os.makedirs(verification_folder, exist_ok=True)

        # Save the image to the verification folder
        image_path = os.path.join(verification_folder, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Image saved successfully at: {image_path}")

        # Preprocess the image and save it in the preprocessed_verification folder
        preprocessed_verification_folder = "C:\\Users\\Smit\\Desktop\\SFDS\\Customized_Dataset\\preprocessed_verification"
        os.makedirs(preprocessed_verification_folder, exist_ok=True)

        # Read the uploaded image using OpenCV's imread function
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # # Resize the image to display it alongside the result
        # resized_uploaded_img = cv2.resize(img, (200, 200))

        # Resize the image to match the input size expected by the model
        resized_img = cv2.resize(img, (128, 128))

        # Normalize the pixel values
        normalized_img = img.astype('float32') / 255.0

        # Save the preprocessed image in the preprocessed_verification folder
        preprocessed_image_path = os.path.join(preprocessed_verification_folder, uploaded_file.name)
        cv2.imwrite(preprocessed_image_path, resized_img * 255)  # Save the preprocessed image

        # Display the uploaded image
        # st.image(resized_uploaded_img, caption="Uploaded Signature", channels="GRAY", use_column_width=True)
        
        # Display the image
        plt.imshow(resized_img, cmap='gray')
        plt.title("Input Signature Image")
        plt.show()
        st.pyplot()

        # Display the prediction result
        prediction = model.predict(np.expand_dims(resized_img, axis=0))
        if prediction < 0.5:
            st.success("The signature is real.")
        else:
            st.error("The signature is forged.")
    else:
        st.warning("Please upload an image before clicking Verify.")
