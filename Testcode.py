from keras.models import load_model
import os
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.io.gfile import GFile
model=load_model(r"C:\Users\SAI SURYA\OneDrive\Desktop\project\plant_leaf_disease_detector.h5")
#model=GFile(r"C:\Users\SAI SURYA\OneDrive\Desktop\project\plant_leaf_disease_detector",'rb')

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


TEST_DIR =r"C:\Users\SAI SURYA\Downloads\New Plant Diseases Dataset Augmented"
TEST_DIR = os.path.join(TEST_DIR, "test")
IMG_SHAPE  = (256, 256)
BATCH_SIZE = 64
from tensorflow.keras.preprocessing.image import load_img,img_to_array, ImageDataGenerator
import numpy as np
test_image_data = []
test_image_filenames = []

for img_name in os.listdir(TEST_DIR):
    img = load_img(os.path.join(TEST_DIR, img_name), target_size = IMG_SHAPE)
    test_image_data.append(img_to_array(img, dtype = 'uint8'))
    test_image_filenames.append(img_name)
    print(img_name)
dicti={0 :' Apple___Apple_scab',1 :' Apple___Black_rot',2 :' Apple___Cedar_apple_rust',3 :' Apple___healthy',4 :' Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',5 :' Corn_(maize)___Common_rust_',6 :' Corn_(maize)___Northern_Leaf_Blight',7 :' Corn_(maize)___healthy',8 :' Potato___Early_blight',9 :' Potato___healthy',10 :' Tomato___Early_blight',11 :' Tomato___Leaf_Mold',12 :' Tomato___Septoria_leaf_spot',13 :' Tomato___Spider_mites Two-spotted_spider_mite',14 :' Tomato___Target_Spot',15 :' Tomato___Tomato_Yellow_Leaf_Curl_Virus',16 :' Tomato___Tomato_mosaic_virus',17 :' Tomato___healthy'}
test_image_data = np.array(test_image_data)/255
print(f'\nTotal testing images: {len(test_image_data)}')
test_pred = np.argmax(model.predict(test_image_data), axis = 1)
for i in test_pred:
    print(i,':'+ dicti[i])

