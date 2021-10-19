from PIL.Image import new
import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im, normalize
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def convolve_im(im, kernel,
                ):
    """ A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    assert len(im.shape) == 3
    #flip the kernel in order to use cross correlation instead
    kernel = np.flipud(np.fliplr(kernel))
    #get the dimensions of the image and kernel
    kernel_y = kernel.shape[0]
    kernel_x = kernel.shape[1]
    im_y = im.shape[0]
    im_x = im.shape[1]
    #create a new blank image with same shape as input
    new_im = np.zeros_like(im)
    #find the center of the kernel
    kernel_center = (int(kernel_y/2), int(kernel_x/2))
    #use zero-padding
    im_padded = np.zeros((im_y + kernel_y - 1, im_x + kernel_x - 1, 3))
    for color in range (3):
        im_padded[kernel_center[0]:-kernel_center[0], kernel_center[1]:-kernel_center[1], color] = im[:,:, color]
    
    #convolve
    for color in range (3):
        for y in range (im_y):
            for x in range (im_x):
                new_im[y, x, color] = np.sum((kernel * im_padded[y:y+kernel_y, x:x+kernel_x, color]))
    return new_im


if __name__ == "__main__":
    # Define the convolutional kernels
    h_b = 1 / 256 * np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Convolve images
    im_smoothed = convolve_im(im.copy(), h_b)
    save_im(output_dir.joinpath("im_smoothed.jpg"), im_smoothed)
    im_sobel = convolve_im(im, sobel_x)
    save_im(output_dir.joinpath("im_sobel.jpg"), im_sobel)

    # DO NOT CHANGE. Checking that your function returns as expected
    assert isinstance(
        im_smoothed, np.ndarray),         f"Your convolve function has to return a np.array. " + f"Was: {type(im_smoothed)}"
    assert im_smoothed.shape == im.shape,         f"Expected smoothed im ({im_smoothed.shape}" + \
        f"to have same shape as im ({im.shape})"
    assert im_sobel.shape == im.shape,         f"Expected smoothed im ({im_sobel.shape}" + \
        f"to have same shape as im ({im.shape})"
    plt.subplot(1, 2, 1)
    plt.imshow(normalize(im_smoothed))

    plt.subplot(1, 2, 2)
    plt.imshow(normalize(im_sobel))
    plt.show()
