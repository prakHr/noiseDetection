import cv2
import numpy as np
'''
# read the image
img = cv2.imread('image.jpg')

# apply the denoiser to the image
denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# calculate the difference between the original and denoised images
diff = cv2.absdiff(img, denoised_img)

# threshold the difference image to obtain the noisy pixels
_, thresholded = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

# convert the difference image to grayscale
gray_diff = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)

# find the coordinates of the noisy pixels
noisy_pixels = np.where(gray_diff > 0)

# print the coordinates of the noisy pixels
print(noisy_pixels)
'''
'''
import skimage.util
from skimage.filters import median

# read the image
img = skimage.io.imread('image.jpg')

# apply the denoiser to the image
denoised_img = median(img, selem=skimage.morphology.square(3))

# calculate the difference between the original and denoised images
diff = skimage.util.img_as_float(img) - skimage.util.img_as_float(denoised_img)

# threshold the difference image to obtain the noisy pixels
thresholded = diff > 0.1

# find the coordinates of the noisy pixels
noisy_pixels = np.where(thresholded)

# print the coordinates of the noisy pixels
print(noisy_pixels)

'''


'''
import cv2

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute the global mean and standard deviation of the grayscale image
mean, stddev = cv2.meanStdDev(gray)

# Create a mask with value 1 for all pixels that are more than 2 standard deviations away from the mean
mask = abs(gray - mean) > 2*stddev

# Use the mask to identify the noisy pixels in the image
noisy_pixels = cv2.bitwise_and(image, image, mask=mask)

'''

# Import necessary libraries
import cv2
import numpy as np

# Read in the input image
# img = cv2.imread('input_image.png')

# Convert the image to grayscale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the grayscale image to remove any high-frequency noise
#blurred = cv2.GaussianBlur(gray, (3,3), 0)

# Use a thresholding technique to detect noisy pixels in the image
# This will likely involve finding the optimal threshold value using a method such as Otsu's thresholding
#threshold, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Visualize the noisy pixels by overlaying them on the original image
#img_with_noise = img.copy()
#img_with_noise[thresholded == 0] = (0, 0, 255)  # set noisy pixels to red

# Save the output image with the noisy pixels highlighted
#cv2.imwrite('output_image.png', img_with_noise)


def uniform_noise(img_path):
    # import numpy as np
    # import scipy
    # from scipy.signal import cwt
    from skimage.io import imread
    # import numpy as np

    # def erlang_wavelet(t, s, k=5):
    #     return np.power(t/s, k) * np.exp(-t/s) / np.sqrt(s * np.math.factorial(k-1))
    # print("here")

    # # Read the input image
    # # img = imread('input_image.png')
    # img = imread(img_path)
    # print('here2')
    # # Perform the continuous wavelet transform (CWT) on the image
    # # using the 'erlang' wavelet
    # # widths = np.arange(1, img.shape[0])

    # # cwt_result = cwt(img, widths, 'erlang')
    # # Define the time variable
    # t = np.linspace(0, 1, img.shape[0], endpoint=False)

    # # Define the scales
    # # scales = np.arange(1, 10)
    # widths = np.arange(1, 31)

    # # Define the shape parameter k
    # # k = 5
    # sig = img.copy()
    # # Apply the CWT with Erlang wavelet function
    # # cwtmatr = sig.cwt(img, erlang_wavelet, scales)
    # cwtmatr = cwt(sig, scipy.signal.ricker, widths)
    # print('here3')

    # # The CWT returns a 2D array, with the first dimension representing the time
    # # (or in this case, the rows of the image) and the second dimension representing
    # # the scale (or columns of the image)
    # # We will use the absolute value of the CWT to identify noisy pixels
    # cwt_result = np.abs(cwtmatr)

    # # Now we will threshold the CWT result to identify noisy pixels
    # # First, we need to determine a suitable threshold value
    # # We can use the median absolute deviation (MAD) to find this value
    # threshold = np.median(cwt_result) + 0.6745 * np.median(np.abs(cwt_result - np.median(cwt_result)))

    # # Now we can threshold the CWT result to identify the noisy pixels
    # noisy_pixels = cwt_result > threshold

    # # The 'noisy_pixels' array will be a boolean array, with 'True'
    # # values indicating the presence of noisy pixels in the image

    # img_with_noise = np.zeros(img.shape)
    # img_with_noise[noisy_pixels] = (0, 0, 255)
    # print("reached here!")
    # return img_with_noise
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as sig
    from skimage import data

    # image = data.coins() # load a sample image from skimage
    import numpy as np
    from functools import lru_cache

    
    def uniform2d_wavelet(omega_x, omega_y):
        # x, y = np.meshgrid(np.arange(-shape // 2, shape // 2 + 1), np.arange(-shape // 2, shape // 2 + 1))
        # r = np.sqrt(omega_x ** 2 + omega_y ** 2)
        # return (r / scale) ** theta * np.exp(-r / scale) / (scale * np.math.factorial(theta - 1))
        return np.ones(omega_x.shape)
    wavelets = {}
    wavelets['uniform'] = uniform2d_wavelet

    

    def _get_wavelet_mask(wavelet: str, omega_x: np.array, omega_y: np.array, **kwargs):
        assert omega_x.shape == omega_y.shape
        return wavelets[wavelet](omega_x, omega_y, **kwargs)

        

    @lru_cache(5)
    def _create_frequency_plane(image_shape: tuple):
        assert len(image_shape) == 2

        h, w = image_shape
        w_2 = (w - 1) // 2
        h_2 = (h - 1) // 2

        w_pulse = 2 * np.pi / w * np.hstack((np.arange(0, w_2 + 1), np.arange(w_2 - w + 1, 0)))
        h_pulse = 2 * np.pi / h * np.hstack((np.arange(0, h_2 + 1), np.arange(h_2 - h + 1, 0)))

        xx, yy = np.meshgrid(w_pulse, h_pulse, indexing='xy')
        dxx_dyy = abs((xx[0, 1] - xx[0, 0]) * (yy[1, 0] - yy[0, 0]))

        return xx, yy, dxx_dyy


    def cwt_2d(x, scales, wavelet, **wavelet_args):
        assert isinstance(x, np.ndarray) and len(x.shape) == 2, 'x should be 2D numpy array'

        x_image = np.fft.fft2(x)
        xx, yy, dxx_dyy = _create_frequency_plane(x_image.shape)

        cwt = []
        wav_norm = []

        for scale_val in scales:
            mask = scale_val * _get_wavelet_mask(wavelet, scale_val * xx, scale_val * yy, **wavelet_args)
            mask = mask.T
            cwt.append(np.fft.ifft2(x_image * mask.T))
            wav_norm.append((np.sum(abs(mask)**2)*dxx_dyy)**(0.5 / (2 * np.pi)))

        cwt = np.stack(cwt, axis=2)
        wav_norm = np.array(wav_norm)

        return cwt, wav_norm

    

    scales = np.arange(1, 10)
    img = cv2.imread(img_path,0)
    # print(img.shape)
    # cwtmatr, freqs = sig.cwt2d(image, erlang2d_wavelet, scales, theta=5, scale=2)
    cwtmatr, freqs = cwt_2d(img,scales,'uniform')
    cwt_result = np.abs(cwtmatr)

    # Now we will threshold the CWT result to identify noisy pixels
    # First, we need to determine a suitable threshold value
    # We can use the median absolute deviation (MAD) to find this value
    threshold = np.median(cwt_result) + 0.6745 * np.median(np.abs(cwt_result - np.median(cwt_result)))

    # Now we can threshold the CWT result to identify the noisy pixels
    noisy_pixels_orig = cwt_result > threshold
    img_with_noises = []
    for i in range(9):
        noisy_pixels = noisy_pixels_orig[:img.shape[0],:img.shape[1],i]
        # The 'noisy_pixels' array will be a boolean array, with 'True'
        # values indicating the presence of noisy pixels in the image
        # img = cv2.imread(img_path,1)
        img_with_noise = np.zeros(img.shape)
        img_with_noise[noisy_pixels] = 255
        img_with_noises.append(img_with_noise)
        # cv2.imwrite(f'img_with_noise-{i}.jpg',img_with_noise)
        # print("reached here!")
    # return img_with_noise
    # return cwtmatr
    return img_with_noises
    
    
    # return cwtmatr
# img_path = r"C:\Users\gprak\Downloads\Github Repos\watermark.png"
# img_path = r"C:\Users\gprak\Downloads\Research Papers\dog\images.jpg"
# img_path = r"C:\Users\gprak\Downloads\Research Papers\label_poisoned_dataset - Copy\aadhar\Aadhaar_letter_large.png"
# img_path = r"C:\Users\gprak\Downloads\Research Papers\lognormal_Noise\Screenshot 2023-03-19 230102.png"
# img_path = r"C:\Users\gprak\Downloads\Research Papers\lena\Screenshot 2023-03-19 223303.png"
# uniform_noise(img_path)
# cv2.imwrite('img_with_noise.jpg',img_with_noise)