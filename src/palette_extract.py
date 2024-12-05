from sklearn.cluster import KMeans
import numpy as np
from .constants import *
from .img_process import ImageProcessor

class PaletteExtractor:
    def __init__(self, processor, n_colors=DEFAULT_NUM_OF_COLORS):
        """
        Initializes the PaletteExtractor class.
        Prepares the KMeans clustering model and the image processor for extracting the color palette.
        
        Args:
        processor: An instance of the ImageProcessor class used to convert LAB colors to RGB.
        n_colors: The number of colors to extract from the image (default is defined in constants).
        """
        self.processor = processor
        self.n_colors = n_colors
        self.kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        self.palette = None
        self.rgb_palette = None

    def fit(self, pixel_values):
        """
        Fits the KMeans model to the pixel values and extracts the color palette.
        
        Args:
        pixel_values: A 2D array of pixel values (each row is a pixel with 3 values for RGB).
        
        Returns:
        self: The fitted PaletteExtractor instance with the extracted color palette.
        """
        self.kmeans.fit(pixel_values)
        self.palette = self.kmeans.cluster_centers_
        return self

    def extract_palette(self):
        """
        Converts the extracted color palette from LAB color space to RGB and returns the RGB values.
        
        Returns:
        rgb_palette: The RGB color palette as a 3D NumPy array.
        
        Raises:
        ValueError: If the palette has not been extracted yet (fit() has not been called).
        """
        if self.palette is None:
            raise ValueError("No palette extracted. Call fit() first.")
        
        lab_palette = np.uint8(self.palette)
        lab_palette_3d = lab_palette.reshape(1, -1, 3)
        self.rgb_palette = self.processor.lab_to_rgb(lab_palette_3d)
        return self.rgb_palette

    def get_hex_palette(self):
        """
        Converts the RGB palette to hexadecimal color codes.
        
        Returns:
        hex_codes: A list of hexadecimal color codes.
        
        Raises:
        ValueError: If the RGB palette has not been extracted yet (extract_palette() has not been called).
        """
        if self.rgb_palette is None:
            raise ValueError("No RGB palette extracted. Call extract_palette() first.")
        
        hex_codes = []
        for batch in self.rgb_palette:
            for rgb in batch:
                hex_codes.append(self.processor.rgb_to_hex(rgb))
        return hex_codes