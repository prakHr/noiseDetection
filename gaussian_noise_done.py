# import necessary libraries
'''
import cv2
import numpy as np
from sklearn.svm import SVC

# load the image and convert it to grayscale
image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply a median filter to the grayscale image to remove high-frequency noise
median = cv2.medianBlur(gray, 5)

# use dilation and erosion to further reduce the noise and improve the image quality
kernel = np.ones((5,5), np.uint8)
dilated = cv2.dilate(median, kernel)
eroded = cv2.erode(dilated, kernel)

# flatten the image into a 1D array and use a support vector machine (SVM) to classify the noisy pixels
flattened = eroded.flatten()
clf = SVC(gamma='auto')
clf.fit(flattened.reshape(-1, 1), np.zeros(flattened.shape[0]))

# predict the class of each pixel in the image
preds = clf.predict(gray.reshape(-1, 1))

# create a binary mask of the noisy pixels in the image
mask = preds.reshape(gray.shape)
mask[mask == 1] = 255
mask[mask == 0] = 0

# apply the mask to the original image to remove the noisy pixels
output = cv2.bitwise_and(image, image, mask=mask)

# save the output image
cv2.imwrite('output.png', output)
'''
'''
import numpy as np
import cv2

# Load the noisy image
img = cv2.imread("noisy_image.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image
blurred = cv2.GaussianBlur(gray, (3,3), 0)

# Compute the absolute difference between the blurred and original images
diff = cv2.absdiff(blurred, gray)

# Threshold the difference image to identify pixels that are significantly different
# from the blurred image (these are the noisy pixels)
thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]

# Identify the noisy pixels in the thresholded image
coords = np.column_stack(np.where(thresh > 0))

# Draw a circle around the noisy pixels
for coord in coords:
    x, y = coord
    cv2.circle(img, (x,y), 5, (0,0,255), -1)

# Save the image with the noisy pixels circled
cv2.imwrite("noisy_pixels_circled.png", img

'''

'''
import numpy as np
from skimage import io, color, filters

# Load the noisy image
img = io.imread("noisy_image.png")

# Convert the image to grayscale
gray = color.rgb2gray(img)

# Apply a Gaussian blur to the image
blurred = filters.gaussian(gray, sigma=3)

# Compute the absolute difference between the blurred and original images
diff = np.abs(blurred - gray)

# Threshold the difference image to identify pixels that are significantly different
# from the blurred image (these are the noisy pixels)
thresh = (diff > 0.1)

# Identify the noisy pixels in the thresholded image
coords = np.column_stack(np.where(thresh))

# Draw a circle around the noisy pixels
for coord in coords:
    x, y = coord
    io.imshow(img)
    io.draw_circle(x, y, 5, color=(0,0,255))

# Save the image with the noisy pixels circled
io.imsave("noisy_pixels_circled.png", img)

'''

'''
import numpy as np
import fastnlmeansdenoising as fnld

# Load the noisy image
img = np.load("noisy_image.npy")

# Apply non-local means denoising to the image
denoised = fnld.fastnlmeansdenoising(img, sigma=3)

# Compute the absolute difference between the denoised and original images
diff = np.abs(denoised - img)

# Threshold the difference image to identify pixels that are significantly different
# from the denoised image (these are the noisy pixels)
thresh = (diff > 0.1)

# Identify the noisy pixels in the thresholded image
coords = np.column_stack(np.where(thresh))

# Draw a circle around the noisy pixels
for coord in coords:
    x, y = coord
    cv2.circle(img, (x,y), 5, (0,0,255), -1)

# Save the image with the noisy pixels circled
np.save("noisy_pixels_circled.npy", img)

'''


'''
# Import necessary libraries
import cv2
import numpy as np
from pywt import dwt2

# Read in the input image
img = cv2.imread('input_image.png')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply the wavelet transform to the grayscale image
coeffs = dwt2(gray, 'db1')

# The coefficients returned by the wavelet transform will contain the noisy pixels
# We can use these coefficients to identify and visualize the noisy pixels in the image
# For example, we can threshold the coefficients to remove any values below a certain threshold
threshold = np.mean(coeffs)
noisy_pixels = np.abs(coeffs) > threshold

# Visualize the noisy pixels by overlaying them on the original image
img_with_noise = img.copy()
img_with_noise[noisy_pixels] = (0, 0, 255)  # set noisy pixels to red

# Save the output image with the noisy pixels highlighted
cv2.imwrite('output_image.png', img_with_noise)

'''

'''
import numpy as np
from scipy.signal import cwt
from skimage.io import imread

# Read the input image
img = imread('input_image.png')

# Perform the continuous wavelet transform (CWT) on the image
# using the 'gaussian' wavelet
cwt_result = cwt(img, 'gaussian')

# The CWT returns a 2D array, with the first dimension representing the time
# (or in this case, the rows of the image) and the second dimension representing
# the scale (or columns of the image)
# We will use the absolute value of the CWT to identify noisy pixels
cwt_result = np.abs(cwt_result)

# Now we will threshold the CWT result to identify noisy pixels
# First, we need to determine a suitable threshold value
# We can use the median absolute deviation (MAD) to find this value
threshold = np.median(cwt_result) + 0.6745 * np.median(np.abs(cwt_result - np.median(cwt_result)))

# Now we can threshold the CWT result to identify the noisy pixels
noisy_pixels = cwt_result > threshold

# The 'noisy_pixels' array will be a boolean array, with 'True'
# values indicating the presence of noisy pixels in the image

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


def gaussian_noise(img_path):
    
    from skimage.io import imread
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as sig
    from skimage import data

    import numpy as np
    from functools import lru_cache

    
    # def gaussian2d_wavelet(omega_x, omega_y, scale, theta):
        # x, y = np.meshgrid(np.arange(-shape // 2, shape // 2 + 1), np.arange(-shape // 2, shape // 2 + 1))
        # r = np.sqrt(omega_x ** 2 + omega_y ** 2)
        # return (r / scale) ** theta * np.exp(-r / scale) / (scale * np.math.factorial(theta - 1))
    def gaus(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1):
        return (1j * omega_x)**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


    def gaus_2(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1):
        return (1j * (omega_x + 1j * omega_y))**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


    def gaus_3(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1, b=1, a=1):
        return (1j * (a * omega_x + b * 1j * omega_y))**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)

    wavelets = {}
    wavelets['gaussian'] = gaus
    wavelets['gaussian2'] = gaus_2
    wavelets['gaussian3'] = gaus_3

    

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
    cwtmatr, freqs = cwt_2d(img,scales,'gaussian')
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
# img_with_noise = gaussian_noise(img_path)
# cv2.imwrite('img_with_noise.jpg',img_with_noise)