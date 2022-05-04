from fastapi import FastAPI,File,UploadFile
import shutil
import os
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, Lambda, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import random
import numpy as np
import pandas as pd
import cv2
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "*"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

alphabets = u"!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
max_str_len = 100 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank(epsilon)
num_of_timestamps = 64 # max length of predicted labels


def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        # find() method returns the lowest index of the substring if it is found in given string otherwise -1

    return np.array(label_num)


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret += alphabets[ch]
    return ret



def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)





from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)



loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


def preprocess(img):
    (h, w) = img.shape

    final_img = np.ones([100, 800]) * 255  # black white image

    # crop
    if w > 800:
        img = img[:, 200:2280]

    if h > 100:
        img = img[300:400, 400:1200]

    final_img[:h, :w] = img
    thresh, image_black = cv2.threshold(final_img, 170, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(image_black, kernel, iterations=1)
    resize_img = cv2.resize(img, (256, 64))

    return resize_img  # cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)



file_path_best = "model.h5"

loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr = 0.0001))

checkpoint = ModelCheckpoint(filepath=file_path_best, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]



#tetbadel



@app.get("/")
async def root():
    return {"message": "Hello World"}


def test():
    ch = "aa"
    return(ch)


@app.post('/test')
async def testt(img: UploadFile=File(...)):
    wiss = test()
    # api_host = 'http://http://127.0.0.1:8000/upload/'
    # headers = {'Content-Type': 'image/jpeg'}
    # image_url = 'http://image.url.com/sample.jpeg'

    # img_file = urllib2.urlopen(image_url)

    # response = requests.post(api_host, data=img_file.read(),
    #   
    #                    headers=headers, verify=False)
    with open('imgs/'+img.filename,"wb") as buffer:
        shutil.copyfileobj(img.file,buffer)

    img_dir = "imgs/"+img.filename
    print("aaaaaaaaaaaaaaaaa")
    image = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    thresh, image_black = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 1), np.uint8)
    c2 = cv2.erode(image_black, kernel, iterations=1)

    image = preprocess(c2)
    image = image / 255
    pred = loaded_model.predict(image.reshape(1, 256, 64, 1))
    decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0])

    output = num_to_label(decoded[0]) + ' '

    var1 = output.split(',')
    number = str(var1[1])
    letter = str(var1[0])

    print(letter)
    return ({"letter":letter,"number":number})