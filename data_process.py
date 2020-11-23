import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
import data_sort
from sklearn.model_selection import train_test_split

PATH = '/home/vedank/Desktop/code/facial_sentiment/dataset/icml_face_data.csv'
ROOT = '/home/vedank/Desktop/code/facial_sentiment/data'

# class names to name folders
class_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

#splitting data into train and test
main_data = pd.read_csv(PATH)
main_data.drop(columns=[' Usage'],inplace =True)
main_x,main_y = main_data.iloc[:,1:],main_data.iloc[:,:1]
x_train,x_test,y_train,y_test = train_test_split(main_x,main_y,test_size=0.25,random_state = 10)

#save labels for future use
np.save(f"{ROOT}/train/y_train.npy",np.array(y_train,dtype=np.uint8))
np.save(f"{ROOT}/test/y_test.npy",np.array(y_test,dtype=np.uint8))

#converts csv data to numpy arrays
def csv_to_np(input_data,ROOT_PATH):
    a = []
    data = pd.Series(np.array(input_data).flatten())
    for i in range(len(data)):  
        img_str = data[i].split(' ')
        image = np.asarray(img_str,dtype=np.uint8).reshape(48,48,1)
        a.append(image)

    data = np.array(a,dtype=np.uint8).squeeze(axis=3)

    data = np.array([np.repeat(image[...,np.newaxis],3,-1)/255.0 for image in data],dtype=np.float32)
    return data 



if __name__ == "__main__":
    for x,y,z in [[x_train,y_train,"train"],[x_test,y_test,"test"]]:

        np_data = csv_to_np(x,ROOT)
        y = np.array(y,dtype=np.uint8)
        data_sort.numpy_to_jpeg_sorted(np_data,y,ROOT,z)

