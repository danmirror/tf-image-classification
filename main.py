import requests
from io import BytesIO

from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pylab as plt
import numpy as np

# # Parameters
input_size = (224,224)

#define input shape
channel = (3,)
input_shape = input_size + channel

#define labels
labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess(img,input_size):
    nimg = img.convert('RGB').resize(input_size, resample= 0)
    img_arr = (np.array(nimg))/255
    return img_arr

def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


MODEL_PATH = 'model'
model = load_model(MODEL_PATH,compile=True)

# read image

im = Image.open('flowers/dandelion/23414449869_ee849a80d4.jpg')
X = preprocess(im,input_size)
X = reshape([X])
y = model.predict(X)

plt.imshow(im)
plt.axis('off')
plt.show()

print( labels[np.argmax(y)], np.max(y) )