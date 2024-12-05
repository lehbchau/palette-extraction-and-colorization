import torch
import numpy as np
import cv2 as cv
from .constants import *
from models.colorizers import siggraph17
from skimage import color
import torch.nn.functional as F

class GrayscaleColorizer:
    def __init__(self, use_gpu=False):
        """
        Initialize colorizer with SIGGRAPH17 pretrained model.
        The model is loaded in evaluation mode.

        Parameters:
            use_gpu (bool): If True, and if a GPU is available, the model is moved to the GPU for faster processing. 
                            Otherwise, it will run on the CPU.
        """
        self.model = siggraph17(pretrained=True).eval()
        if use_gpu and torch.cuda.is_available():
            self.model.cuda()
        else:
            self.use_gpu = False

    def load_img(self, uploaded_img):
        """
        Load the image, converting it to RGB if it is grayscale.
        If the uploaded image is a grayscale image, it is converted into RGB.
        """
        # Convert the uploaded image from bytes to a numpy array
        img_bytes = uploaded_img.getvalue()
        img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)

        # Decode the image using OpenCV
        img = cv.imdecode(img_array, cv.IMREAD_UNCHANGED)

        # If grayscale, convert to 3 channels
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        return img

    def resize_img(self, img, HW=(COLORIZE_SIZE), resample=cv.INTER_LINEAR):
        """
        Resize the input image to a target height and width (HW).
        Default resample method is INTER_LINEAR (bilinear interpolation).
        """
        return cv.resize(img, (HW[1], HW[0]), interpolation=resample)

    def preprocess_img(self, img_rgb_orig, HW=(COLORIZE_SIZE)):
        """
        Preprocess the image for colorization.
        - Resizes the input image.
        - Converts it to LAB color space and extracts the L (lightness) channel.
        - Returns the L channel for the original and resized images as tensors.
        """
        img_rgb_rs = self.resize_img(img_rgb_orig, HW=HW)
        
        img_lab_orig = color.rgb2lab(img_rgb_orig)
        img_lab_rs = color.rgb2lab(img_rgb_rs)

        img_l_orig = img_lab_orig[:, :, 0]
        img_l_rs = img_lab_rs[:, :, 0]

        tens_orig_l = torch.Tensor(img_l_orig)[None, None, :, :]
        tens_rs_l = torch.Tensor(img_l_rs)[None, None, :, :]

        return (tens_orig_l, tens_rs_l)

    def postprocess_tens(self, tens_orig_l, out_ab):
        """
        Postprocess the output from the colorization model.
        - The output 'ab' (color) channels are resized to match the original L channel.
        - The L and ab channels are combined to form the final LAB image.
        - Convert LAB to RGB color space.
        """
        HW_orig = tens_orig_l.shape[2:] # Original image dimensions
        HW = out_ab.shape[2:] # Output image dimensions from the model

        # If dimensions differ, resize the 'ab' channels to match the original dimensions
        if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
            out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
        else:
            out_ab_orig = out_ab

        out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
        return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))

    def colorize(self, rgb_img):
        """
        Colorize a grayscale image.
        - Preprocess the image.
        - Pass it through the SIGGRAPH17 model for colorization.
        - Postprocess the output and return the colorized image.
        """
        # Load and preprocess image
        tens_l_orig, tens_l_rs = self.preprocess_img(rgb_img, HW=(COLORIZE_SIZE))

        if self.use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        # Colorize using SIGGRAPH17 model
        out_img_siggraph17 = self.postprocess_tens(tens_l_orig, self.model(tens_l_rs).cpu())
        return out_img_siggraph17