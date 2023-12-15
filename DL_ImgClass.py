# necessary imports
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model_path = './model_tools.tf'
model = load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = img_array
    return img, preprocessed_img

# Function to make predictions
def predict_image(image_array, model):
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)
    probabilities = predictions[0]
    return predicted_class[0], probabilities

# Streamlit UI
st.title('Task DL: Image Classification - Joppe Vanhelden')

# Sidebar for uploading image
uploaded_file = st.sidebar.file_uploader("Choose a JPG image", type="jpg")
SEARCH_TERMS = ["hammer", "pliers", "handsaw", "screwdriver", "wrench"]

# Display the uploaded image in the sidebar
if uploaded_file is not None:
    img, preprocessed_image = preprocess_image(uploaded_file)
    st.sidebar.image(img, caption='Uploaded Image', use_column_width=True)

# images of classes
st.subheader('The different classes:')
columns = st.columns(5)
for i, class_name in enumerate(SEARCH_TERMS):
    sample_image_path = f'./images/Streamlit/{class_name}.jpg'
    columns[i].image(sample_image_path, caption=f'{class_name.capitalize()} Image', use_column_width=True)

# Main section with EDA
st.subheader('EDA (Exploratory Data Analysis)')
st.markdown("The category I chose is tools, with these five tools: hammer, handsaw, pliers, screwdriver, and wrench.")
st.write("The image below describes my data split, I splitted my data into 3 sets: training dataset, validation dataset, and a testing dataset. Here you can see how the data is divided over these 3 sets.")

# Placeholder for EDA image
eda_image_path = './images/Streamlit/EDA.png'
st.image(eda_image_path, caption='EDA Image', use_column_width=True)

# Confusion matrices section
st.subheader('Comparison of Confusion Matrices')
st.markdown("Here you can see the confusion matrices for my model (left) and Google's Teachable Machine model (right). One thing is very clear when looking at this, Google's Teachable Machine model is very good and has almost zero flaws.")
st.markdown("Compared to Google's Teachable Machine model, my model has a lot of flaws and doesn't really do a very good job with predicting what class a picture belongs to. One reason for this is that my model will never even come close to the one from Google. Another reason may be that my categories aren't that easy to predict and might look the same to my model.")

# Placeholder for confusion matrices
custom_matrix_path = './images/Streamlit/conf_matrix.png'
teachable_machine_matrix_path = './images/Streamlit/GTM_confmatrix.png'

columns = st.columns(2)
columns[0].image(custom_matrix_path, caption="Custom Model Confusion Matrix", use_column_width=True)
columns[1].image(teachable_machine_matrix_path, caption="Teachable Machine Confusion Matrix", use_column_width=True)

# Prediction section
if uploaded_file is not None:
    # Preprocess the image and make predictions
    predicted_class, probabilities = predict_image(preprocessed_image, model)

    # Display the predicted class and probabilities beneath the EDA and confusion matrices
    st.subheader(f'Prediction: {SEARCH_TERMS[predicted_class]}')
    
    st.subheader('Probabilities:')
    for i, class_prob in enumerate(probabilities):
        # Convert probability to percentage
        probability_percentage = class_prob * 100
        st.write(f'{SEARCH_TERMS[i]}: {class_prob:.4f} ({probability_percentage:.2f}%)')