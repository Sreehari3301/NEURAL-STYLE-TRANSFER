# Neural Style Transfer (NST) with VGG19

This project implements **Neural Style Transfer** using TensorFlow and the VGG19 model. Neural Style Transfer is an optimization technique used to take two images—a *content image* and a *style reference image* (such as an artwork by a famous painter)—and blend them together so the output image looks like the content image, but “painted” in the style of the style reference image.

## How it Works
1. **Feature Extraction**: We use a pre-trained VGG19 network to extract representational features of both images.
2. **Content Loss**: We minimize the difference between the intermediate feature maps of the content image and the generated image.
3. **Style Loss**: We calculate the Gram Matrix of the intermediate feature maps to capture the "texture" or "style" of the style image.
4. **Total Variation Loss**: We include a small penalty for high-frequency noise to ensure the resulting image is smooth.
5. **Optimization**: We use the Adam optimizer to update the pixels of the generated image to minimize the total loss.

## Files
- `nst_vgg19.py`: The main Python script containing the implementation.
- `content.png`: The original photograph (generated for this demo).
- `style.png`: The artistic style image (generated for this demo).
- `stylized_image.png`: The final output (generated after running the script).

## Requirements
- `tensorflow`
- `numpy`
- `pillow`
- `matplotlib`

### 1. Prerequisites
Ensure you have **Python 3.10 or higher** installed on your system.

### 2. Setup Environment
It is recommended to use a virtual environment to keep dependencies organized:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Running the Application

#### **A. Web Interface (Recommended)**
To run the interactive dashboard where you can upload and convert images:
```bash
python app.py
```
Then, open your browser and go to: `http://127.0.0.1:5000`

#### **B. Command Line Script**
To run the high-quality script directly:
1. Place your images as `content.png` and `style.png` in the project folder.
2. Run the script:
   ```bash
   python nst_vgg19.py
   ```

## 🛠️ Project Structure
- `app.py`: Flask backend server for the web interface.
- `index.html`: Modern frontend for uploading and viewing results.
- `nst_vgg19.py`: Core logic for Neural Style Transfer using VGG19.
- `requirements.txt`: List of required Python packages.
- `uploads/`: Directory where your uploaded images are stored.
- `results/`: Directory where the stylized masterpieces are saved.
