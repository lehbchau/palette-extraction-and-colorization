import json
import csv
import io
import numpy as np
from .img_process import ImageProcessor

class PaletteExporter:
    def __init__(self, processor):
        """
        Initialize the PaletteExporter with an ImageProcessor instance.
        
        Args:
            processor (ImageProcessor): An instance of the ImageProcessor class.
        """
        self.processor = processor

    def _encode_np(self, obj):
        """Helper function to handle np types while serializing to JSON."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def save_palette_json(self, colors, filename='palette.json'):
        """
        Save the color palette as a JSON file.

        Args:
            colors (np.ndarray): The color palette in RGB format.
            filename (str): The name of the JSON file to save the palette.

        Returns:
            str: JSON string of the color palette.
        """
        palette = {}
        for batch in colors:
            for i, color in enumerate(batch):
                palette[f'color_{i+1}'] = {
                    'RGB': tuple(color),
                    'Hex': self.processor.rgb_to_hex(color),
                    'HSL': self.processor.rgb_to_hsl(color),
                    'HSV': self.processor.rgb_to_hsv(color),
                    'CMYK': self.processor.rgb_to_cmyk(color)
                }

        palette_json = json.dumps(palette, indent=4, default=self._encode_np)
        return palette_json

    def save_palette_csv(self, colors, filename='palette.csv'):
        """
        Save the color palette as a CSV file.

        Args:
            colors (np.ndarray): The color palette in RGB format.
            filename (str): The name of the CSV file to save the palette.

        Returns:
            str: CSV string of the color palette.
        """
        csv_buff = io.StringIO()
        writer = csv.writer(csv_buff)
        writer.writerow(['Color', 'Hex', 'RGB', 'HSL', 'HSV', 'CMYK'])

        for batch in colors:
            for color in batch:
                hex_code = self.processor.rgb_to_hex(color)
                hsl = self.processor.rgb_to_hsl(color)
                hsv = self.processor.rgb_to_hsv(color)
                cmyk = self.processor.rgb_to_cmyk(color)
                color_int = tuple(map(int, color))
                writer.writerow([color, hex_code, color_int, hsl, hsv, cmyk])

        palette_csv = csv_buff.getvalue()
        csv_buff.close()
        return palette_csv

