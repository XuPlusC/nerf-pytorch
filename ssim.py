from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import math


# img_origin = np.array(Image.open('/home/rec/Workspace/Python/nerf-pytorch/data/nerf_llff_data/orchids/images_8/image000.png'))
img_origin = np.array(Image.open('/home/rec/Workspace/Python/nerf-pytorch/data/nerf_llff_data/horns/images_8/DJI_20200223_163016_842.png'))

# img_t = np.array(Image.open('/home/rec/Workspace/Python/nerf-pytorch/logs/fern_truth/testset_050000/000.png'))
img_t = np.array(Image.open('/home/rec/Workspace/Python/nerf-pytorch/logs/xr_horns/testset_050000/000.png'))


def psnr2(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == "__main__":
	# If the input is a multichannel (color) image, set multichannel=True.
    print("ssim:") 
    print(ssim(img_origin, img_t, multichannel=True))
    print("psnr:")
    print(psnr2(img_origin, img_t))
