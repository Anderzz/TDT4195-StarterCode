from numpy.fft import fft
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
    # START YOUR CODE HERE ### (You can change anything inside this block)
    fft_im = np.fft.fft2(im)

    #create a filter in the freq domain
    #and shift it to line up
    filter = np.ones_like(fft_im)
    locy = len(filter)//2
    locx = len(filter[1])//2
    filter[locy-5:locy+5,:] = 0
    filter[:,locx-10:locx+10] = 1
    filter_shifted=np.fft.fftshift(magnitude(filter))

    #apply the filter
    im_filtered_fft = fft_im * filter_shifted
    viz_im = np.fft.fftshift(np.log(magnitude(fft_im)+1))
    im_filtered = np.fft.ifft2(im_filtered_fft).real
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    #visualize the filter
    plt.imshow(filter.real, cmap="gray")
    plt.subplot(1, 2, 2)
    #visualize the result
    plt.imshow(im_filtered, cmap="gray")
    plt.show()
    ### END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(im_filtered))
