

# Brain Tumor Analysis Application

## Overview
The **Brain Tumor Analysis** application leverages deep learning techniques to classify and detect brain tumors in medical imaging. The application provides two main functionalities: **Tumor Classification** and **Tumor Detection**. It uses a **Convolutional Neural Network (CNN)** for classification and **YOLOv5** for detecting tumors in axial, coronal, and sagittal MRI scans. With an accuracy of **99.4%** in classification, this tool offers a highly reliable solution for brain tumor analysis.


---

## Features

- **Tumor Classification**: Classify the tumor type (Glioma, Meningioma, No Tumor, Pituitary) using a pre-trained TensorFlow model.
- **Tumor Detection**: Detect tumors in axial, coronal, and sagittal planes using YOLOv5-based models.
- **User-Friendly Interface**: Simple and clean interface built with Tkinter to make the process as smooth as possible.

---

## Requirements

To run this application, ensure you have the following installed:

- Python 3.x
- Tkinter
- TensorFlow
- PyTorch
- Pillow (for image processing)
- YOLOv5 model weights

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/PerceptiaAI/brain-tumor-classification-and-detection.git
    
    cd brain-tumor-classification-and-detection
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the necessary YOLOv5 model weights and TensorFlow checkpoint and place them in the appropriate folders (check the `gui.py` for exact paths).

---

## How to Use

### 1. Launch the Application
To start the application, simply run the `gui.py` file:
```bash
python gui.py
```

### 2. Select the Mode
Upon opening the application, you can choose between two modes:
- **Classify Tumor**: Classifies the tumor type in a selected image.
- **Detect Tumor**: Detects and highlights the tumor in an image, showing it in the selected plane (axial, coronal, sagittal).

### 3. Upload Your Image
Click on the **"Select Image"** button to choose an MRI scan image from your local drive (only .jpg, .png, or .jpeg formats supported).

### 4. Process the Image
Once you’ve uploaded the image:
- For **Classification**, click **"Process"** to get the predicted tumor class.
- For **Detection**, choose the desired plane (Axial, Coronal, or Sagittal) and click **"Process"** to detect and highlight the tumor.

---

## GUI Overview

### Main Window
- **Buttons**:
  - **Classify Tumor**: Selects the tumor classification mode.
  - **Detect Tumor**: Switches to tumor detection mode.
  
- **Plane Selection (for Detection)**: Choose between **Axial**, **Coronal**, or **Sagittal** to view tumor detection in different views.

- **Image Display**: Shows the image you’ve uploaded or the processed result.

- **Result Label**: Displays the results of the classification or detection.

### Icons and Images
For better clarity, the interface uses small icons to represent different brain planes (Axial, Coronal, Sagittal) and a clean, minimalistic layout to ensure ease of use.

---

## Code Explanation

### `gui.py` File
The core functionality of the application resides in `gui.py`. This file uses **Tkinter** for the graphical interface, **TensorFlow** for tumor classification, and **PyTorch** (with YOLOv5) for tumor detection.

- **Image Preprocessing**: Images are resized and normalized for both classification and detection tasks.
- **Model Loading**: The classification model is loaded using TensorFlow, and the detection models (one for each plane) are loaded using PyTorch.
- **Detection and Classification**: Based on the selected mode, the program either classifies the tumor or detects it within the image, with results shown in the GUI.

---

## Contribution

If you would like to contribute to the development of this project, feel free to fork the repository and make changes. You can create pull requests for any improvements or bug fixes.

---

## Acknowledgments

- **YOLOv5**: For the tumor detection models.
- **TensorFlow**: For the tumor classification model.
- **Tkinter**: For the GUI framework.
- **Pillow**: For image handling and manipulation.

---

**Enjoy using Brain Tumor Analysis, and happy coding!**

--- 
