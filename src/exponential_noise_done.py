import cv2
import numpy as np
'''
# Compute the inverse of the pixel values in the input image
inv_image = 1/image

# Convert the inverted image to grayscale
gray = cv2.cvtColor(inv_image, cv2.COLOR_BGR2GRAY)

# Compute the global mean and standard deviation of the grayscale image
mean, stddev = cv2.meanStdDev(gray)

# Create a mask with value 1 for all pixels that are more than 2 standard deviations away from the mean
mask = abs(gray - mean) > 2*stddev

# Use the mask to identify the noisy pixels in the inverted image
noisy_pixels_inv = cv2.bitwise_and(inv_image, inv_image, mask=mask)

# Convert the noisy pixels in the inverted image back to the original scale
noisy_pixels = 1/noisy_pixels_inv
'''

def exponential_noise(img):
    # import numpy as np
    # import scipy
    # from scipy.signal import cwt
    from skimage.io import imread
    
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.signal as sig
    from skimage import data

    # image = data.coins() # load a sample image from skimage
    import numpy as np
    from functools import lru_cache

    
    def exponential2d_wavelet(omega_x, omega_y, tau):
        # x, y = np.meshgrid(np.arange(-shape // 2, shape // 2 + 1), np.arange(-shape // 2, shape // 2 + 1))
        r = np.sqrt(omega_x ** 2 + omega_y ** 2)
        kernel = np.exp(-r / tau)
        return kernel
    wavelets = {}
    wavelets['exponential'] = exponential2d_wavelet

    

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
    # img = cv2.imread(img_path,0)
    # img = img_path.copy()
    # print(img.shape)
    # cwtmatr, freqs = sig.cwt2d(image, erlang2d_wavelet, scales, theta=5, scale=2)
    cwtmatr, freqs = cwt_2d(img,scales,'exponential',tau=10)
    cwt_result = np.abs(cwtmatr)

    # Now we will threshold the CWT result to identify noisy pixels
    # First, we need to determine a suitable threshold value
    # We can use the median absolute deviation (MAD) to find this value
    threshold = np.median(cwt_result) + 0.6745 * np.median(np.abs(cwt_result - np.median(cwt_result)))

    # Now we can threshold the CWT result to identify the noisy pixels
    noisy_pixels_orig = cwt_result > threshold
    # print(noisy_pixels.shape)
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
# exponential_noise(img_path)
# cv2.imwrite('img_with_noise.jpg',img_with_noise)