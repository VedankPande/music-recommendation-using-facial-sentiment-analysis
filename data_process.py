import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
import data_sort

TRAIN_PATH = '/home/vedank/Desktop/code/facial_sentiment/dataset/train.csv'
TEST_PATH = '/home/vedank/Desktop/code/facial_sentiment/dataset/test.csv'
ROOT = '/home/vedank/Desktop/code/facial_sentiment/data'

# class names to name folders
class_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

#converts csv data to numpy arrays
def csv_to_np(PATH,ROOT_PATH,dataset_type):
    a = []
    data = pd.read_csv(PATH)
    if dataset_type == "train":
        data,data_y = data.iloc[:,1:],data.iloc[:,:1]
        np.save(f"{ROOT_PATH}/train_data_labels.npy",np.array(data_y))
    data = pd.Series(np.array(data).flatten())
    for i in range(len(data)):  
        img_str = data[i].split(' ')
        image = np.asarray(img_str,dtype=np.uint8).reshape(48,48,1)
        a.append(image)

    data = np.array(a,dtype=np.uint8).squeeze(axis=3)

    data = np.array([np.repeat(image[...,np.newaxis],3,-1)/255.0 for image in data],dtype=np.float32)
    return data 



if __name__ == "__main__":
    np_train = csv_to_np(TRAIN_PATH,ROOT,"train")
    label = np.load(f"{ROOT}/train_data_labels.npy")
    data_sort.numpy_to_jpeg_sorted(np_train,label,ROOT)

