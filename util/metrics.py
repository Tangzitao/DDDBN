import numpy as np
import math
from skimage.metrics import structural_similarity as ssim

def SSIM(img1, img2):
	# ssim_val = ssim(img1, img2, data_range=1.0, gaussian_weights=True, use_sample_covariance=False, multichannel=True, channel_axis=2)
	# 这里要指定data_range和channel_axis=2(哪个是通道维)
	ssim_val = ssim(img1, img2, gaussian_weights=True, use_sample_covariance=False, multichannel=True)
	return ssim_val
	
def PSNR(img1, img2):
	mse = np.mean( (img1 - img2) ** 2 )
	if mse == 0:
		return 100
	PIXEL_MAX = 1
	return 10 * math.log10(PIXEL_MAX / mse)
