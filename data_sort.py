import numpy as np
from PIL import Image

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

# takes numpy arrays and saves them in jpeg format, in class based folders
def numpy_to_jpeg_sorted(train,labels,ROOT_PATH,train_or_test):

    class_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    
    class_count = {emote:0 for emote in class_names}

    for idx,image in enumerate(train):
        emote = labels[idx]
        emote_name = class_names[int(emote)]
        class_count[emote_name] += 1
        
        # saving in jpeg format so the pytorch dataloader.ImageFolder can read the files
        im = Image.fromarray((image*255.0).astype(np.uint8))
        im.save(f"{ROOT_PATH}/{train_or_test}/{class_names[int(emote)]}/{class_count[emote_name]}.jpeg")

        #uncomment below if you want .npy files instead
        #np.save(f"{ROOT_PATH}/{class_names[int(emote)]}/{class_count[emote_name]}.npy",np.array(image))

    print(class_count)
