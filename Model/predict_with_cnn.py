from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import math


def crop_image_by_pixel(x, y, image):
    im = Image.open(rf"{image}")
    y = math.floor(y)
    x = math.floor(x)
    left = y - 40
    up = x - 40
    right = y + 41
    down = x + 41
    cropped_im = im.crop((left, up, right, down))

    return cropped_im


def prediction(current_frame, auxiliary):

    loaded_model = load_model(r"C:\Users\Shirel\Documents\אקסלנטים\bootcamp\mobileye_integration\Model\model.h5")
    for color in ['red', 'green']:
        cur_auxiliary = auxiliary[color]
        crops_lst = [np.array(crop_image_by_pixel(aux[0], aux[1], current_frame)).tolist() for aux in cur_auxiliary]

        l_predictions = loaded_model.predict(crops_lst)
        l_predicted_label = np.argmax(l_predictions, axis=-1)
        auxiliary[color] = [x for i, x in enumerate(cur_auxiliary) if l_predicted_label[i] == 1]
    return auxiliary
