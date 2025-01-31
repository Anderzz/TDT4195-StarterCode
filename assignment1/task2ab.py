import matplotlib.pyplot as plt
import pathlib
from utils import read_im, save_im
output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """  
    return 0.212*im[:,:,0] + 0.7152*im[:,:,1] + 0.0722*im[:,:,2]


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")
plt.show()


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
        max(map(max, im))
    """
    if max(x.any() for x in im)>1:
        return 255-im
    else:
        return 1-im

plt.imshow(inverse(im_greyscale), cmap="gray")
plt.show()
