import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils

def magnitude(fft_im):
    real = fft_im.real
    imag = fft_im.imag
    return np.sqrt(real**2 + imag**2)

def convolve_im(im: np.array,
                kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the spatial kernel (kernel),
        and returns the resulting image.

        "verbose" can be used for turning on/off visualization
        convolution

        Note: kernel can be of different shape than im.

    Args:
        im: np.array of shape [H, W]
        kernel: np.array of shape [K, K] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)

    #since kernel might be of different shape than the image
    #we need to pad with the difference
    x_pad = len(im[1])-len(kernel[1])
    y_pad = len(im[0])-len(kernel[0])
    kernel = np.pad(kernel, ((0, y_pad), (x_pad, 0)), 'constant', constant_values=(0,0))

    #apply fft to both the kernel and image
    im_fourier = np.fft.fft2(im)
    kernel_fourier = np.fft.fft2(kernel)
    #use the convolution theorem
    conved_im_fourier = im_fourier * kernel_fourier

    #shift DC component to center and apply log transform for visualization
    viz_kernel_fft = np.fft.fftshift(np.log(magnitude(kernel_fourier)+1))
    viz_im = np.fft.fftshift(np.log(magnitude(im_fourier)+1))
    viz_im_conved = np.fft.fftshift(np.log(magnitude(conved_im_fourier)+1))

    #shift back to spatial domain with inverse fft
    conv_result = np.fft.ifft2(conved_im_fourier).real

    if verbose:
        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(20, 4))
        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")
        plt.subplot(1, 5, 2)
        # Visualize FFT
        plt.imshow(viz_im, cmap="gray")
        plt.subplot(1, 5, 3)
        # Visualize FFT kernel
        plt.imshow(viz_kernel_fft, cmap="gray")
        plt.subplot(1, 5, 4)
        # Visualize filtered FFT image
        plt.imshow(viz_im_conved, cmap="gray")
        plt.subplot(1, 5, 5)
        # Visualize filtered spatial image
        plt.imshow(conv_result, cmap="gray")

    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True  # change if you want

    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)

    # DO NOT CHANGE
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256
    image_gaussian = convolve_im(im, gaussian_kernel, verbose)

    # DO NOT CHANGE
    sobel_horizontal = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    image_sobelx = convolve_im(im, sobel_horizontal, verbose)

    if verbose:
        plt.show()

    utils.save_im("camera_gaussian.png", image_gaussian)
    utils.save_im("camera_sobelx.png", image_sobelx)
