import numpy as np
from PIL import Image
import scipy.ndimage.filters as filters
from scipy.ndimage.filters import maximum_filter
from scipy import signal as sg
import scipy.ndimage as nd


def find_tfl(c_image: np.ndarray):
    c_image = np.array((Image.open(c_image)))
    r_scale, g_scale, b_scale = Image.Image.split(Image.fromarray(c_image))

    # kernal
    kernal27 = np.array(Image.open(r'C:\Users\Shirel\Documents\אקסלנטים\bootcamp\mobileye_integration\Model\kernal27.png'))
    real_kernel = (kernal27 - kernal27.mean()) / 1e5  # kernel normalization
    real_kernel_r = real_kernel[:, :, 0]
    real_kernel_g = real_kernel[:, :, 1]
    real_kernel_b = real_kernel[:, :, 2]

    # red_channel
    r_scale = np.array(r_scale)
    image_red_after_convolve_e = sg.convolve(r_scale, real_kernel_r, mode="same")
    image_max_red = filters.maximum_filter(image_red_after_convolve_e, 300)
    red_lst = np.where(image_red_after_convolve_e == image_max_red)

    # green_channel
    g_scale = np.array(g_scale)
    image_green_after_convolve_e = sg.convolve(g_scale, real_kernel_g, mode="same")
    image_max_green = filters.maximum_filter(image_green_after_convolve_e, 300)
    green_lst = np.where(image_green_after_convolve_e == image_max_green)

    # [y,x]
    auxiliary = {'red': list(zip(red_lst[0], red_lst[1])),
                 'green': list(zip(green_lst[0], green_lst[1]))}

    return auxiliary
