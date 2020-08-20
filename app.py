import streamlit as st
from keras.models import model_from_json
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import keras
from PIL import Image, ImageOps
html_temp1 = """
    <div style="background-color:black;padding:4px">
    <p style="color:white;text-align:center;">ABOUT</p>
    </div>
    <p>--------------------------------------------------</p>
    <p> This is a web app for identification of salt in the image plz upload the test image in the dataset provided by kaggle .</p>
    <p>TGS SALT</p>
    """
html_temp2 = """
    <p>Instructions:</p>
    <p> Result after uploading image seismic image is for better visualization of the uploaded image and salt predicted is the result where salt is found marked in yellow</p>
    """
st.sidebar.markdown(html_temp1,unsafe_allow_html=True)
img_file  = open("samv.png","rb").read()
st.sidebar.markdown(html_temp2,unsafe_allow_html = True)
st.sidebar.image(img_file)    
st.title("TGS SALT IDENTIFICATION")
st.header("USING KERAS AND TENSORFLOW")
# load json and create model
def preprocess_test_image(path):
    im_width = 128
    im_height = 128
    X = np.zeros((1, im_height, im_width, 1), dtype=np.float32)
    img = load_img(path,grayscale=True)
    x_img = img_to_array(img)
    x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    X[0] = x_img/255.0
    return X
def predict(path):
    json_file = open('tgsmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("tgsmodel.h5")
    print("Loaded model from disk")
    img_array = preprocess_test_image(path)
    preds_val_dum = model.predict(img_array, verbose=1)
    preds_val_dum = (preds_val_dum > 0.5).astype(np.uint8)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(img_array[0,...,0], cmap='seismic')
    ax[0].set_title('Original Image converted into seismic')
    ax[0].axis('off')
    ax[1].imshow(preds_val_dum[0,...,0], vmin=0, vmax=1)
    ax[1].set_title('Salt Predicted');
    plt.axis('off')
    st.pyplot()
st.set_option('deprecation.showfileUploaderEncoding', False)
uploaded_file = st.file_uploader("Choose Image")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded IMage.',width=300)
    st.write("")
    st.write("Predicting")
    predict(uploaded_file)
    st.success('prediction Sucessfull')


    

    