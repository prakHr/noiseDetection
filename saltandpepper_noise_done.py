# Import the PIL package
from PIL import Image

# Open the image file
# img = Image.open("image.jpg")

# Apply the median filter
# img = img.filter(ImageFilter.MedianFilter)

# Save the filtered image
# img.save("filtered_image.jpg")

'''
import cv2

# Load the image with salt and pepper noise
image = cv2.imread("salt_pepper_noise.png")

# Create a DnCNN model and load pre-trained weights
model = DnCNN()
model.load_weights("dncnn_weights.h5")

# Use the model to denoise the image
denoised_image = model.predict(image)

# Find the noisy pixels by comparing the denoised and original images
noisy_pixels = np.where(image != denoised_image)

'''

'''
import cv2

# read the image
img = cv2.imread('image.jpg')

# apply the denoiser to the image
denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# save the denoised image
cv2.imwrite('denoised_image.jpg', denoised_img)

'''
def saltandpepper_noise(img_path):
	import cv2
	import numpy as np
	import pywt

	img = cv2.imread(img_path, 0)
	def cwt(image):
	    coeffs2 = pywt.dwt2(image, 'haar')
	    LL, (LH, HL, HH) = coeffs2
	    return LL, LH, HL, HH

	LL, LH, HL, HH = cwt(img)

	LH_thr = cv2.threshold(np.abs(LH), 0.8*np.std(np.abs(LH)), 255, cv2.THRESH_BINARY)[1]
	HL_thr = cv2.threshold(np.abs(HL), 0.8*np.std(np.abs(HL)), 255, cv2.THRESH_BINARY)[1]
	HH_thr = cv2.threshold(np.abs(HH), 0.8*np.std(np.abs(HH)), 255, cv2.THRESH_BINARY)[1]

	mask = cv2.bitwise_or(cv2.bitwise_or(LH_thr, HL_thr), HH_thr)
	mask = cv2.resize(mask,(img.shape[1],img.shape[0]))
	# print(f"{mask.shape} and {img.shape} mask={mask}")
	# img = np.float32(img)
	# mask = np.float32(mask)
	# img = cv2.cvtColor(img, cv2.CV_8U)
	# noisy_pixels = cv2.cvtColor(mask, cv2.CV_8U)
	# noisy_pixels = mask
	# img_with_noise = cv2.bitwise_and(img, mask)
	# noisy_pixels = img & mask
	# noisy_pixels = cv2.bitwise_and(img, img, mask=mask)
	# print("Reached here!")
	img_with_noises = []
	img_with_noise = np.zeros(img.shape)
	# img_with_noise[noisy_pixels] = 255
	img_with_noise = 255.-mask
	img_with_noises.append(img_with_noise)
	return img_with_noises

# img_path = r"C:\Users\gprak\Downloads\Github Repos\barcode.jpg"
# salt_and_pepper_noise(img_path)