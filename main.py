import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Path to user database (JSON file)
user_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

# Load the model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "C:/Users/Swapnil Mhaske/Desktop/testcspro2/plant-disease-prediction-cnn-deep-leanring-project-main/model_training_notebook/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to load user data from the JSON file
def load_user_db():
    if os.path.exists(user_db_path):
        with open(user_db_path, "r") as f:
            return json.load(f)
    else:
        return {}

# Function to save user data to the JSON file
def save_user_db(user_db):
    with open(user_db_path, "w") as f:
        json.dump(user_db, f)

# Function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict image class
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence_score = predictions[0][predicted_class_index]
    return predicted_class_name, confidence_score

# Function to display the login page
def login():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Login to Plant Disease Detection</h2>", unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")

    if st.button("Login", use_container_width=True):
        user_db = load_user_db()
        if username in user_db and user_db[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password")

# Function to display the signup page
def signup():
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Sign Up for Plant Disease Detection</h2>", unsafe_allow_html=True)

    new_username = st.text_input("Choose a Username", placeholder="Enter a new username")
    new_password = st.text_input("Choose a Password", type="password", placeholder="Enter a new password")
    confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")

    if st.button("Sign Up", use_container_width=True):
        user_db = load_user_db()
        if new_username in user_db:
            st.error("Username already exists. Please choose a different username.")
        elif new_password != confirm_password:
            st.error("Passwords do not match. Please try again.")
        elif new_username == "" or new_password == "":
            st.error("Username and password cannot be empty.")
        else:
            user_db[new_username] = new_password
            save_user_db(user_db)
            st.success(f"Account created successfully for {new_username}! Please log in.")
            st.session_state['show_login'] = True

# Function to display the header with logout
def display_header():
    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>🌿 Plant Disease Detection</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>Welcome, {st.session_state['username']}!</h4>", unsafe_allow_html=True)
    
    if st.button("Logout", key="logout_button", use_container_width=True):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ""
        st.session_state['show_login'] = False
        # Immediately re-render the app after logout
        st.session_state['logout'] = True

# Function to display the main plant disease detection app
def plant_disease_detection():
    st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")
    
    if 'logout' in st.session_state and st.session_state['logout']:
        st.session_state['logout'] = False
        return  # Skip rendering the main content to allow the login/signup options to display

    display_header()
    st.subheader("Upload an image to classify the plant disease")

    uploaded_image = st.file_uploader("Choose an image file (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.markdown("### Uploaded Image Preview:")
        image = Image.open(uploaded_image).resize((500, 500))
        st.image(image, caption="Uploaded Image", use_column_width=False)

        if st.button("🔍 Classify Plant Disease", use_container_width=True):
            prediction, confidence_score = predict_image_class(model, uploaded_image, class_indices)

            # Create a more interactive display for the results
            col1, col2 = st.columns([2, 1])  # Adjust column widths

            with col1:
                st.markdown(f"## 🌱 **Prediction**: **{prediction}**")
                st.markdown(f"### Confidence Score: **{confidence_score:.2%}**")

                # Add a confidence score message
                if confidence_score > 0.7:
                    st.success("High confidence in prediction!")
                elif confidence_score > 0.5:
                    st.warning("Moderate confidence in prediction.")
                else:
                    st.error("Low confidence in prediction. Please check the image or try again.")

            with col2:
                st.markdown("### Suggested Actions:")
                st.markdown("- Check the plant for visible symptoms.")
                st.markdown("- Consult with a horticulturist for proper diagnosis.")
                st.markdown("- Research possible treatment options online.")

    else:
        st.warning("Please upload an image file to get started.")

    st.markdown("""
    ---
    *Built with 💚 by Swapnil Mhaske*
    """)

# Main app logic
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'show_login' not in st.session_state:
    st.session_state['show_login'] = False

# Navigation for login and signup
if not st.session_state['logged_in']:
    if st.session_state['show_login']:
        login()
    else:
        menu_option = st.sidebar.selectbox("Menu", ["Login", "Sign Up"])
        if menu_option == "Login":
            login()
        elif menu_option == "Sign Up":
            signup()
else:
    plant_disease_detection()
