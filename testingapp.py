import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model

# EPOCHS = 10
# INIT_LR = 1e-3
# BS = 16
default_image_size = tuple((256, 256))
image_size = 0
width = 256
height = 256
depth = 3
print('Loading Model')
model = load_model(r"C:\Users\SAI SURYA\OneDrive\Desktop\project\plant_leaf_disease_detector.h5")
print('Model Loaded')
# disease_list = ['Bacterial Blight', 'Blast', 'Brown Spot', 'Healthy']


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        label.config(image=image)
        label.image = image

        predict_disease(file_path)


def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img, (256, 256))

    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img


def predict(image):
    test_pred = model.predict(image)
    # probabilities = model.predict(np.asarray([image]))[0]
    # class_idx = np.argmax(probabilities)
    return test_pred
disease_list={0 :' Apple___Apple_scab',1 :' Apple___Black_rot',2 :' Apple___Cedar_apple_rust',3 :' Apple___healthy',4 :' Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',5 :' Corn_(maize)___Common_rust_',6 :' Corn_(maize)___Northern_Leaf_Blight',7 :' Corn_(maize)___healthy',8 :' Potato___Early_blight',9 :' Potato___healthy',10 :' Tomato___Early_blight',11 :' Tomato___Leaf_Mold',12 :' Tomato___Septoria_leaf_spot',13 :' Tomato___Spider_mites Two-spotted_spider_mite',14 :' Tomato___Target_Spot',15 :' Tomato___Tomato_Yellow_Leaf_Curl_Virus',16 :' Tomato___Tomato_mosaic_virus',17 :' Tomato___healthy'}


def predict_disease(image_path):

    img = load_image(image_path)
    # print(img.shape())
    pr = predict(img)

    prediction = np.argmax(pr)
    # percentage = pr[prediction]
    # percentage = percentage*100
    predicted_disease = disease_list[prediction]
    print(predicted_disease)

    # Update the label with the predicted disease
    # percentage=round(percentage, 2)
    prediction_label.config(text=f'Predicted Disease: {predicted_disease}')


root = tk.Tk()
root.title('Disease Prediction')
root.geometry('1400x800')

root.configure(bg='#EFEFEF')
root.option_add('*Font', 'Helvetica 12')

image = tk.PhotoImage(file="C:/Users/SAI SURYA/Downloads/mini.png")
background_label = tk.Label(root, image=image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

image_frame = tk.Frame(root, bg='#EFEFEF', width=400, height=400)
image_frame.pack(pady=20)

label = tk.Label(image_frame)
label.pack()

#prediction_frame = tk.Frame(root, bg='#068944')
#prediction_frame.pack()

prediction_label = tk.Label(root, text='Plant Leaf Disease Detection', font=(
    'Helvetica', 16), bg='#068944', fg='white')
prediction_label.pack(pady=10)

upload_button = tk.Button(root, text='Insert Image', command=open_image, font=(
    'Helvetica', 14), bg='#06cacd', fg='black', activebackground='#0056b3', activeforeground='white')
upload_button.pack(pady=10, padx=20)

exit_button = tk.Button(root, text='Exit', command=root.quit, font=(
    'Helvetica', 14), bg='#dc3545', fg='black', activebackground='#b02a37', activeforeground='white')
exit_button.pack(pady=10, padx=20)

root.mainloop()
