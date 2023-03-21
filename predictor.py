import numpy as np
# from keras.preprocessing import image
import keras.utils as image
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
saved_model = load_model("model/VGG_model.h5",compile=False)
status = True


def check(input_img):
    print(" your image is : " + input_img)
    print(input_img)

    img = image.load_img("images/" + input_img, target_size=(224, 224))
    img = np.asarray(img)
    print(img)

    img = np.expand_dims(img, axis=0)

    print(img)
    output = saved_model.predict(img)

    print(output)
    if round(output[0][0],2) == 1:
        status = True
    else:
        status = False

    print(status)
    return status
