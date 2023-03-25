import numpy as np
import cv2

def speckle_noise(image, threshold=1.5):
    # image = cv2.imread(img_path,0)
    # image = img_path.copy()
    # Calculate the mean and standard deviation of the image
    mean, std = np.mean(image), np.std(image)
    
    # Calculate the threshold value based on the provided threshold coefficient
    thresh = mean + threshold * std
    
    # Create a mask of noisy pixels by comparing each pixel to the threshold value
    noisy_pixels_mask = image > thresh
    
    img_with_noise = np.zeros(image.shape)
    img_with_noise[noisy_pixels_mask] = 255

    return [img_with_noise]

# Load the image into a NumPy array
# img_path = r"C:\Users\gprak\Downloads\Github Repos\watermark.png"
# img_path = r"C:\Users\gprak\Downloads\Research Papers\dog\images.jpg"
# img_path = r"C:\Users\gprak\Downloads\Research Papers\label_poisoned_dataset - Copy\aadhar\Aadhaar_letter_large.png"
# img_path = r"C:\Users\gprak\Downloads\Research Papers\label_poisoned_dataset - Copy\invoice\Screenshot 2023-03-19 223303.png"
# img_path = r"C:\Users\gprak\Downloads\Research Papers\lena\Screenshot 2023-03-19 223303.png"

# Detect the noisy pixels in the image
# noisy_pixels_mask = speckle_noise(img_path, threshold=1.5)
# cv2.imwrite('img_with_noise.jpg',noisy_pixels_mask)

# Apply the mask to the original image to get a new image with the noisy pixels removed
# filtered_image = image * noisy_pixels_mask

'''
import numpy as np
from sklearn.covariance import LedoitWolf

def detect_noisy_pixels(image: np.ndarray) -> np.ndarray:
    # Use the Ledoit-Wolf method to estimate the covariance matrix of the image
    lw_cov = LedoitWolf().fit(image)
    cov = lw_cov.covariance_

    # Calculate the mean and standard deviation of the image
    mean, std = np.mean(image), np.std(image)
    
    # Calculate the threshold value based on the estimated covariance matrix
    thresh = mean + cov * std
    
    # Create a mask of noisy pixels by comparing each pixel to the threshold value
    noisy_pixels_mask = image > thresh
    
    return noisy_pixels_mask
'''