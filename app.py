import numpy as np
import streamlit as st
import pandas as pd
import numpy as pd
import sklearn
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from PIL import Image


#######################################
# Functions to Process the Image Input
#######################################

# function to check if the image is related to the task at hand
def get_predictions_confidence(model, img):
    # setting the threshold value
    threshold = 0.8
    probabilities = model.predict_proba(img)
    # finding the class with the highest probability
    predicted_class = np.argmax(probabilities)
    predicted_probability = probabilities[np.arange(len(probabilities)), predicted_class]
    if predicted_probability > threshold:
        return img
    else:
        st.warning('Please upload an image related to the task', icon="⚠")


# function to preprocess the uploaded image
def pre_processing_image(loaded_imaged, image_size):
    img = load_img(loaded_imaged, target_size=image_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    img_array = img_array / 255.0
    return img_array


# function make predictions using the logistic regression model
def make_predictions(logistic_model, test_data, label_encoder):
    predictions = logistic_model.predict(test_data)
    predictions = label_encoder.inverse_transform(predictions)
    return predictions


# function to display the accuracy_score and the predicted
def display_predictions(predictions):
    results = predictions[0]
    st.write('Defect Detected:')
    st.text(results)


# initialising the RandomForestClassifier
random_model_path = 'rf_classifier.sav '
random_model = joblib.load(random_model_path)

label_encoder_path = 'label_encoder.sav '
label_encoder = joblib.load(label_encoder_path)

# loading the VGG16 Model
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# ensuring the VGG16 layers are untrainable
for layer in VGG_model.layers:
    layer.trainable = False


#############################
# creating the Streamlit app
#############################

def main():
    st.title('Metal Surface Defects Detection')
    image = Image.open('karan-bhatia-ib7jwp7m0iA-unsplash.jpg')
    st.image(image, caption='Steel Industry', width=450)
    st.subheader('Detect Defect Present', divider=True)
    st.markdown('The Prediction model will focus on greyscale images that may '
                'have the following surface defects:')
    lst = ['Rolled-in scale', 'Patches', 'Crazing', 'Pitted Surface', 'Inclusion', 'Scratches']
    space = ''
    for defect in lst:
        space += '- ' + defect + "\n"
    st.markdown(space)

    # user uploads an image
    uploaded_img = st.file_uploader('Upload a Photo')
    file_size = (224, 224)
    if uploaded_img is not None:
        # preprocessing the uploaded image
        img_array = pre_processing_image(uploaded_img, file_size)

        # performing feature extraction
        features = VGG_model.predict(img_array)
        features = features.flatten()

        # displaying the uploaded image
        st.subheader('Uploaded Image')
        st.image(uploaded_img, use_column_width=True)
        trial_results = get_predictions_confidence(random_model, features.reshape(1, -1))
        if trial_results is not None:
            # getting the predictions and displaying the results
            predictions = make_predictions(random_model, features.reshape(1, -1), label_encoder)
            display_predictions(predictions)

    else:
        st.warning('Please upload an Image to proceed!')


if __name__ == '__main__':
    main()
