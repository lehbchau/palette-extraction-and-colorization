import cv2 as cv
import numpy as np
from .constants import *

class ImageProcessor:
    def __init__(self, uploaded_img):
        """
        Initializes the ImageProcessor class.
        Loads, resizes, and preprocesses the uploaded image for further processing.
        
        Args:
        uploaded_img: The uploaded image as a streamlit file object.
        """
        self.uploaded_img = uploaded_img
        self.img = self.load_uploaded_img(uploaded_img)
        self.resized_img = self.resize_img(self.img)
        self.lab_img = self.bgr_to_lab(self.img)
        self.blurred_img = self.gaussian_blur(self.lab_img)
        self.pixels = self.flatten_data(self.blurred_img)

    def load_uploaded_img(self, uploaded_img):
        """
        Loads the uploaded image from the streamlit file object and converts it to a NumPy array.
        
        Args:
        uploaded_img: The uploaded image as a streamlit file object.
        
        Returns:
        img: The image loaded as a NumPy array.
        """
        img_bytes = uploaded_img.getvalue()
        img_array = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv.imdecode(img_array, 1)
        return img

    def resize_img(self, img):
        """
        Resizes the image to a target size while maintaining the aspect ratio.
        
        Args:
        img: The image to resize.
        
        Returns:
        img: The resized image.
        """
        height, width = img.shape[:2]
        new_height = height
        new_width = width
        
        if (height > width):
            img_ratio = width / height
            new_height = TARGET_RESIZE
            new_width = int(new_height * img_ratio)

        else:
            img_ratio = height / width
            new_width = TARGET_RESIZE
            new_height = int(new_width * img_ratio)

        return cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)

    def bgr_to_lab(self, bgr):
        """
        Converts a BGR image to LAB color space.
        
        Args:
        bgr: The BGR image to convert.
        
        Returns:
        lab: The converted LAB image.
        """
        return cv.cvtColor(bgr, cv.COLOR_BGR2LAB)

    def bgr_to_rgb(self, bgr):
        """
        Converts a BGR image to RGB color space.
        
        Args:
        bgr: The BGR image to convert.
        
        Returns:
        rgb: The converted RGB image.
        """
        return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    def lab_to_rgb(self, lab):
        """
        Converts a LAB image to RGB color space.
        
        Args:
        lab: The LAB image to convert.
        
        Returns:
        rgb: The converted RGB image.
        """
        return cv.cvtColor(lab, cv.COLOR_LAB2RGB)

    def rgb_to_hex(self, rgb):
        """
        Converts an RGB color to hexadecimal format.
        
        Args:
        rgb: The RGB color as a tuple (R, G, B).
        
        Returns:
        hex_color: The hexadecimal color as a string.
        """
        return '#{:02X}{:02X}{:02X}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def rgb_to_hsl(self, rgb):
        """
        Converts an RGB color to HSL (Hue, Saturation, Lightness) format.
        
        Args:
        rgb: The RGB color as a tuple (R, G, B).
        
        Returns:
        hsl: The HSL color as a tuple (H, S, L).
        """
        # Normalize RGB values
        r, g, b = rgb[0] / RGB_SCALE, rgb[1] / RGB_SCALE, rgb[2] / RGB_SCALE

        # Find max and min RGB values
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val

        # Calculate Lightness 
        l = (max_val + min_val) / 2

        # Calculate Saturation
        if diff == 0:
            s = 0
        else:
            if l <= 0.5:
                s = diff / (max_val + min_val)
            else:
                s = diff / (2.0 - max_val - min_val)

        # Calculate Hue
        if diff == 0:  # No saturation (Grayscale)
            h = 0
        elif max_val == r:
            h = ((g - b) / diff) % 6
        elif max_val == g:
            h = 2.0 + (b - r) / diff
        else:
            h = 4.0 + (r - g) / diff 

        h = h * 60  # Convert to degrees
        if h < 0:
            h += 360

        return (round(h), round(s * 100), round(l * 100))

    def rgb_to_hsv(self, rgb):
        """
        Converts an RGB color to HSV (Hue, Saturation, Value) format.
        
        Args:
        rgb: The RGB color as a tuple (R, G, B).
        
        Returns:
        hsv: The HSV color as a tuple (H, S, V).
        """
        # Normalize RGB values
        r, g, b = rgb[0] / RGB_SCALE, rgb[1] / RGB_SCALE, rgb[2] / RGB_SCALE  

        # Find max and min RGB values
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val

        # Calculate Hue 
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff)) % 360  
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360  
        elif max_val == b:
            h = (60 * ((r - g) / diff) + 240) % 360 

        # Calculate Saturation 
        if max_val == 0:
            s = 0  
        else:
            s = (diff / max_val) * 100  

        # Calculate Value 
        v = max_val * 100  

        return (round(h), round(s), round(v))  

    def rgb_to_cmyk(self, rgb):
        """
        Converts an RGB color to CMYK (Cyan, Magenta, Yellow, Key/Black) format.
        
        Args:
        rgb: The RGB color as a tuple (R, G, B).
        
        Returns:
        cmyk: The CMYK color as a tuple (C, M, Y, K).
        """
        # Normalize RGB values
        r, g, b = rgb[0] / RGB_SCALE, rgb[1] / RGB_SCALE, rgb[2] / RGB_SCALE
        
        # Find the maximum and minimum values of the RGB
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val

        # Calculate Black Key (K)
        k = 1 - max_val
        
        # If K is not 1 (pure black), calculate C, M, Y
        if k != 1:  
            c = (1 - r - k) / (1 - k)
            m = (1 - g - k) / (1 - k)
            y = (1 - b - k) / (1 - k)

        # If K is 1 (pure black), C, M, Y are 0
        else:  
            c = 0
            m = 0
            y = 0

        return round(c * 100), round(m * 100), round(y * 100), round(k * 100)

    def gaussian_blur(self, img):
        """
        Applies Gaussian blur to the image to reduce noise and details.
        
        Args:
        img: The image to apply Gaussian blur on.
        
        Returns:
        blurred_img: The image after applying Gaussian blur.
        """
        return cv.GaussianBlur(img, (3,3), 0)

    def flatten_data(self, img):
        """
        Flattens the image data into a 2D array for easier processing (e.g., for clustering or other algorithms).
        
        Args:
        img: The image to flatten.
        
        Returns:
        flattened_img: The flattened image data.
        """
        return img.reshape((-1, 3))

    def preprocess_img(self):
        """
        Preprocess the image and return the flattened pixel data.
        
        Returns:
        pixels: The preprocessed pixel data from the blurred image.
        """
        return self.pixels