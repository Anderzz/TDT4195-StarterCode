import numpy as np
from numpy.core.fromnumeric import argmax, mean
from numpy.lib.function_base import average
import skimage
import utils
import pathlib
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

def cum_mean(arr):
    cum_sum = np.cumsum(arr, axis=0)    
    for i in range(cum_sum.shape[0]):       
        if i == 0:
            continue        
        #print(cum_sum[i] / (i + 1))
        cum_sum[i] =  cum_sum[i] / (i + 1)
    return cum_sum


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Uses the recipe on p751 in the book to 
        find a threshold using Otsu's method

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    #find the bins
    bins = np.arange(257)
    i = np.arange(256)
    #compute normalized histogram
    p, _ = np.histogram(im, bins=bins, density=True)
    #compute cumulative sums
    P1 = lambda k: np.sum(p[:k])
    #compute cumulative means
    cum_mean = lambda k: np.sum(i[:k]*p[:k])
    #compute the global mean
    MG = np.sum(i*p)
    #compute between-class variance
    sigmasq = lambda k: (MG*P1(k)-cum_mean(k))**2/(P1(k)*(1-P1(k))) if P1(k) > 0 else 0
    #find the threshold
    threshold = np.average(np.argmax([sigmasq(k) for k in i]))
    #check the against the real solution
    otsu_real = threshold_otsu(im)
    print("correct value:", otsu_real)
    return threshold
    ### END YOUR CODE HERE ###

if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)
