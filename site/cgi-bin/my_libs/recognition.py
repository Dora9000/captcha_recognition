#!/usr/bin/env python
# coding: utf-8

import keras
import pickle
import os
import cv2 

def recognition_1():
    CUR_PATH = os.getcwd()
    CUR_PATH = CUR_PATH[:-4]
    PATH = CUR_PATH + "/pic/"
    CUR_PATH += "/cgi-bin/my_libs/"
    model = keras.models.load_model(CUR_PATH + '/network/model_25.h5')
    lb = pickle.loads(open(CUR_PATH  + "/network/bin_class.txt", "rb").read()) #загружаем сохраненный бинаризатор меток

    os.chdir(PATH)
    listdir = os.listdir()
    answer = ""
    for letter in listdir:
        if 'img' in letter:
            continue
        if 'jpeg' not in letter and 'png' not in letter :
            continue
        if '.' in letter:
            #print(PATH + letter)
            test_image = cv2.imread(PATH + letter)
            test_image = cv2.resize(test_image,(28,28))
            test_image = test_image.astype("float") / 255.0
            test_image = test_image.reshape((1,test_image.shape[0], test_image.shape[1],test_image.shape[2]))
            preds = model.predict(test_image)
            i = preds.argmax(axis=1)[0]
            test_label = lb.classes_[i]
            #print("Результат распознавания: " + test_label)
            answer += test_label
    return answer

def recognition_2():
    CUR_PATH = os.getcwd()
    CUR_PATH = CUR_PATH[:-4]
    PATH = CUR_PATH + "/pic/"
    CUR_PATH += "/cgi-bin/my_libs/"
    model = keras.models.load_model(CUR_PATH + '/network/2/model_2_26.h5')
    lb = pickle.loads(open(CUR_PATH  + "/network/2/bin_class.txt", "rb").read()) #загружаем сохраненный бинаризатор меток

    os.chdir(PATH)
    listdir = os.listdir()
    answer = ""
    for letter in listdir:
        if 'img' in letter:
            continue
        if 'jpeg' not in letter and 'png' not in letter :
            continue
        if '.' in letter:
            #print(PATH + letter)
            test_image = cv2.imread(PATH + letter)
            test_image = cv2.resize(test_image,(28,28))
            test_image = test_image.astype("float") / 255.0
            test_image = test_image.reshape((1,test_image.shape[0], test_image.shape[1],test_image.shape[2]))
            preds = model.predict(test_image)
            i = preds.argmax(axis=1)[0]
            test_label = lb.classes_[i]
            #print("Результат распознавания: " + test_label)
            answer += test_label
    return answer