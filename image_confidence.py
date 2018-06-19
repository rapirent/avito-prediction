from keras.preprocessing import image
from PIL import Image
import keras.applications.inception_v3 as inception_v3
import pandas as pd
import numpy as np
import os

print('confidence encodeing start...')

image_files = [x.path for x in os.scandir('./data/images/')]
inception_model = inception_v3.InceptionV3(weights='imagenet')

def classify_inception(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
    except (OSError, IOError):
        return [0,0,0.5]
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception_v3.preprocess_input(x)
    preds = inception_model.predict(x)
    return inception_v3.decode_predictions(preds, top=1)[0][0]

def image_id_from_path(path):
    return path.split('/')[3].split('.')[0]

inception_conf = [[image_id_from_path(x), classify_inception(x)[2]] for x in image_files]
confidence = pd.DataFrame(inception_conf, columns=['image', 'image_confidence'])
confidence.to_csv('./encoded_image.csv')
