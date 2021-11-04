import skimage
import skimage.io
import skimage.transform
import os
import numpy as np
import utils
import matplotlib.pyplot as plt

def magnitude(fft_im):
    real = fft_im.real
    imag = fft_im.imag
    return np.sqrt(real**2 + imag**2)


if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)
    #plt.imshow(im, cmap="gray")
    #plt.show()
    # START YOUR CODE HERE ### (You can change anything inside this block)
    im_filtered = im
    fft_im = np.fft.fft2(im)
    frequency_kernel_low_pass = utils.create_low_pass_frequency_kernel(im, radius=50)
    #fft_im = fft_im*frequency_kernel_low_pass
    viz_im = np.fft.fftshift(np.log(magnitude(fft_im)+1))
    im_filtered = np.fft.ifft2(fft_im).real 
    plt.imshow(viz_im, cmap="gray")
    plt.show()
    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(im_filtered))
