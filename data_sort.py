import os
import numpy as np

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

x_train = np.load('/home/vedank/Desktop/code/facial_sentiment/npy_data/train_data.npy')
labels = np.load('/home/vedank/Desktop/code/facial_sentiment/npy_data/train_data_labels.npy')
test =  np.load('/home/vedank/Desktop/code/facial_sentiment/npy_data/test_data.npy')

ROOT_PATH = '/home/vedank/Desktop/code/facial_sentiment/data'

class_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

class_count = {emote:0 for emote in class_names}

for idx,image in enumerate(x_train):
    emote = labels[idx]
    emote_name = class_names[int(emote)]
    class_count[emote_name] += 1
    # since np.save does not create a new directory automatically, you'll have to manually create folders for each class
    np.save(f"{ROOT_PATH}/{class_names[int(emote)]}/{class_count[emote_name]}.npy",np.array(image))

print(class_count)
