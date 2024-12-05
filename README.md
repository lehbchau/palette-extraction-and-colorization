# AI-Powered Color Palette Extraction and Image Colorization

A **Streamlit** app showcasing **AI and Machine Learning** techniques for advanced image processing tasks:  
- Image Color Palette Extraction using K-Means Clustering
- Grayscale Image Colorization using a Pretrained Neural Network (SIGGRAPH17)

## Features

### 1. Image Color Palette Extraction
- Upload an image to extract its **dominant color palette** using **K-Means Clustering**, a classic unsupervised machine learning algorithm.
- **OpenCV** is used for preprocessing, while **Scikit-learn** powers the clustering of pixel colors.
- Adjustable slider to extract between **4 to 15 dominant colors**, visualized with **Matplotlib**.
- Each color is presented with its **Hex code** and exportable to **JSON** or **CSV**, containing:
  - RGB, Hex, HSL, HSV, and CMYK color representations.

### 2. Grayscale Image Colorization
- Upload a grayscale image, and **neural network-based AI** brings it to life in color.
- Utilizes the **SIGGRAPH '17 pretrained model**, a **deep learning model trained on a large image dataset** for realistic and context-aware colorization.
- Combines the power of **PyTorch** and **Skimage** for preprocessing and postprocessing.
- Displays the **original grayscale** and **colorized image side by side** for seamless comparison.

## How the AI Works
- **Color Palette Extraction**:
  - Uses **unsupervised learning (K-Means)** to identify clusters in pixel color data, producing an accurate and concise representation of dominant colors in an image.
- **Grayscale Image Colorization**:
  - Built on the **SIGGRAPH '17 model**, which leverages a **convolutional neural network (CNN)** to predict color channels from grayscale images, offering vibrant and natural results.
  - This AI model was trained on an extensive dataset to generalize well across various image types and styles.

## How to Use
### Online
1. Visit the app on Streamlit Cloud: [Try the app here](#replace-with-your-link).  
2. Follow the intuitive instructions within the app interface.

### Local Setup (Optional)
1. Clone this repository:
   ```bash
   git clone https://github.com/lehbchau/palette-extraction-and-colorization.git
   cd palette-extraction-and-colorization\
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app
   ```bash
   streamlit run main.py
   ```

## Technologies Used
- **Python**
- **Streamlit** for web interface
- **OpenCV** for image processing
- **Scikit-learn** for clustering colors
- **Matplotlib** for palette visualization
- **Torch** for deep learning
- **Pretrained SIGGRAPH17 model** for grayscale image colorization
- **Skimage** for grayscale image processing before colorization in LAB and RGB color spaces

## Demo

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
It also uses a pretrained model from the Colorization Repository, licensed under its original terms. The license for the pretrained models can be found in models/LICENSE.

## Acknowledgements
This project uses the SIGGRAPH17 colorization model, developed by Richard Zhang, Phillip Isola, and Alexei A. Efros. This pretrained model was used for colorizing grayscale images. The model and source code can be found [here](https://github.com/richzhang/colorization).
