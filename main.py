import streamlit as st
import numpy as np
from src.img_process import ImageProcessor
from src.palette_extract import PaletteExtractor
from src.export import PaletteExporter
from src.visualize import visualize_palette
from src.colorize import *

# Sidebar for navigation
st.sidebar.title("Features")
feature = st.sidebar.radio(
    "Choose a feature:", 
    ("Color Palette Extraction", "Grayscale Image Colorization")
)

# Reset session state when switching features
if 'uploaded_img' in st.session_state:
    uploaded_img = st.session_state['uploaded_img']
else:
    uploaded_img = None

if 'current_feature' not in st.session_state:
    st.session_state['current_feature'] = feature
elif st.session_state['current_feature'] != feature:
    st.session_state['current_feature'] = feature
    if 'uploaded_img' in st.session_state:
        del st.session_state['uploaded_img']

# App title
st.title("Palette Extractor & Image Colorizer")
st.markdown("_Discover dominant colors and breathe life into grayscale images._")

if feature == "Color Palette Extraction":
    # Feature header
    st.header("Extract Color Palette From Your Image")

    st.markdown("**How to use?** :sunglasses:")
    instruction = '''
    1. Upload your image (JPG, JPEG, PNG)
    
    2. Adjust number of colors in palette as needed (default to 7)

    3. View extracted palette (Hex codes included)

    4. Export palette to CSV or JSON (supporting RGB, Hex, HSL, HSV, CMYK)
    
    '''
    st.markdown(instruction)

    # File uploader
    uploaded_img = st.file_uploader(
            label="Upload your image here", 
            type=['jpg', 'png', 'jpeg'],
            key="uploaded_img"
        )

    # Check if an image is uploaded
    if uploaded_img is not None:
        # Display success message
        st.success("Image uploaded successfully!", icon="✅")

        # Create an instance of the ImageProcessor class
        processor = ImageProcessor(uploaded_img)

        # Preprocess the image
        raw_pixels = processor.preprocess_img()

        # Select number of colors displayed in palette
        num_colors = st.slider("Select number of colors in palette", min_value=4, max_value=15, value=7, step=1)
        st.write("Your color palette currently has ", num_colors, "colors")

        with st.spinner("Extracting palette. Please wait..."):
            # Create an instance of the PaletteExtractor class
            palette_extractor = PaletteExtractor(processor, num_colors).fit(raw_pixels)

            # Extract the color palette
            rgb_palette = palette_extractor.extract_palette()

            # Display the uploaded image
            st.image(uploaded_img, caption='Your Image', use_container_width=True)

            # Visualize the color palette
            palette_fig = visualize_palette(rgb_palette, palette_extractor)
            st.pyplot(fig=palette_fig, use_container_width=True)

        # Options to export palette to JSON/CSV files
        col1, _, col2 = st.columns([2.5, 9, 2.5])

        exporter = PaletteExporter(processor)
        palette_csv = exporter.save_palette_csv(rgb_palette)
        palette_json = exporter.save_palette_json(rgb_palette)

        with col1:
            st.download_button(
                label="Export CSV",
                data=palette_csv,
                file_name="palette.csv",
                icon=":material/download:",
                mime="text/csv"
            )

        with col2:
            st.download_button(
                label="Export JSON",
                data=palette_json,
                file_name="palette.json",
                icon=":material/download:",
                mime="text/json"
            )

if feature == "Grayscale Image Colorization":
    # Feature header
    st.header("Colorize Your Grayscale Image")

    st.markdown("**How to use?** :sunglasses:")
    instruction = '''
    1. Upload your grayscale image (JPG, JPEG, PNG)
    
    2. Get colorized image
    
    '''
    st.markdown(instruction)

    # File uploader
    uploaded_img = st.file_uploader(
            label="Upload your image here", 
            type=['jpg', 'png', 'jpeg'],
            key="uploaded_img"
        )

    # Check if an image is uploaded
    if uploaded_img is not None:
        # Display success message
        st.success("Image uploaded successfully!", icon="✅")

        # Start colorizing process
        with st.spinner("Colorizing. Please wait..."):
            colorizer = GrayscaleColorizer()
            img = colorizer.load_img(uploaded_img)

            # Extract the original L channel from the image and resize 
            (tens_l_orig, tens_l_rs) = colorizer.preprocess_img(img, HW=(256,256))
            img_bw = colorizer.postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))
            output_img = colorizer.postprocess_tens(tens_l_orig, colorizer.model(tens_l_rs).cpu())

        # Create 2 columns to display before and after images
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Grayscale")
            st.image(img_bw)

        with col2:
            st.subheader("Color")
            st.image(output_img)