import matplotlib.pyplot as plt
import numpy as np
from .palette_extract import PaletteExtractor
from .constants import *

def visualize_palette(rgb_palette, palette_extractor):
    """
    Create a visualization of the color palette as a Matplotlib figure.

    Parameters:
        rgb_palette (ndarray): A 3D array of shape (1, n_colors, 3) containing RGB color values.
        clustering_instance (PaletteExtractor): An instance of the PaletteExtractor class.

    Returns:
        fig (Figure): A Matplotlib figure containing the palette visualization.
    """
    # Get number of colors in the palette
    n_colors = rgb_palette.shape[1]  
    bar_width = BAR_WIDTH
    bar_height = BAR_HEIGHT
    hex_palette = palette_extractor.get_hex_palette()

    # Create a blank canvas for palette visualization
    img_width = n_colors * bar_width
    palette_img = np.zeros((bar_height, img_width, 3), dtype=np.uint8)

    # Fill the canvas with palette colors
    for i in range(n_colors):
        start_x = i * bar_width
        end_x = start_x + bar_width
        color = rgb_palette[0, i]  # Extract color from palette
        palette_img[:, start_x:end_x, :] = color

    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(palette_img)
    ax.axis("off")

    # Determine font size based on number of colors in palette
    if n_colors < 6:
        font_size = 16
    elif n_colors < 10:
        font_size = 14
    elif n_colors < 13:
        font_size = 10
    else:
        font_size = 8

    # Add HEX color codes as text below the bars
    for i, hex_code in enumerate(hex_palette):
        ax.text(
            x=(i + 0.5) * bar_width, 
            y=bar_height + 12, 
            s=hex_code, 
            ha='center', 
            va='bottom', 
            fontsize=font_size, 
            color='k'
        )

    plt.tight_layout(pad=4.0)
    plt.close(fig)
    return fig