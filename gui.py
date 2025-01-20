import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tensorflow as tf
import torch
import numpy as np
import sys
from pathlib import Path

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

# Ensure YOLOv5 is in the Python path
sys.path.append('yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# Load classification model
classification_model = tf.keras.models.load_model('tumor_model_checkpoint.keras')

# Define class mappings
class_mappings = {0: 'Glioma', 1: 'Meninigioma', 2: 'No tumor', 3: 'Pituitary'}

# Load detection models
device = select_device('')
detection_models = {
    'axial': attempt_load(r'P:\Projects\brain tumor danial\Final draft\output_models\tumor_detector_axial.pt'),
    'coronal': attempt_load(r'P:\Projects\brain tumor danial\Final draft\output_models\tumor_detector_coronal.pt'),
    'sagittal': attempt_load(r'P:\Projects\brain tumor danial\Final draft\output_models\tumor_detector_sagittal.pt')
}

# Preprocess image for classification
def preprocess_image(image):
    image = image.resize((168, 168))
    image = np.array(image.convert('L'))  # Convert to grayscale
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Classification prediction
def predict_class(image):
    preprocessed_image = preprocess_image(image)
    prediction = classification_model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    return class_mappings[predicted_class]

# Detect tumors using YOLOv5
def detect_tumor(image_path, plane):
    model = detection_models[plane]
    
    # Load original image
    original_img = Image.open(image_path)
    
    # Prepare image for model
    img_size = 640
    img = original_img.copy()
    img.thumbnail((img_size, img_size))
    
    # Create a new image with padding
    new_image = Image.new("RGB", (img_size, img_size), (114, 114, 114))
    new_image.paste(img, ((img_size - img.width) // 2, (img_size - img.height) // 2))

    # Convert to numpy array
    img_array = np.array(new_image)
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Inference
    pred = model(img_tensor)[0]
    pred = non_max_suppression(pred, 0.6, 0.45)

    # Process detections
    result_img = original_img.copy()
    draw = ImageDraw.Draw(result_img)
    
    # Calculate scaling factors
    scale_x = original_img.width / img_size
    scale_y = original_img.height / img_size
    
    for det in pred:
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                # Scale coordinates to original image size
                x1, y1, x2, y2 = [int(coord * scale_x) if i % 2 == 0 else int(coord * scale_y) for i, coord in enumerate(xyxy)]
                
                c = int(cls)
                label = f'{model.names[c]} {conf:.2f}'
                draw_box_and_label(draw, [x1, y1, x2, y2], label)

    return result_img

# Draw bounding box and label on the image
def draw_box_and_label(draw, xyxy, label):
    xyxy = [int(x) for x in xyxy]
    draw.rectangle(xyxy, outline="red", width=3)
    font = ImageFont.truetype("arial.ttf", 16)
    bbox = font.getbbox(label)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.rectangle([xyxy[0], xyxy[1] - text_height - 4, xyxy[0] + text_width, xyxy[1]], fill="red")
    draw.text((xyxy[0], xyxy[1] - text_height - 4), label, fill="white", font=font)

# GUI Class
class BrainTumorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Brain Tumor Analysis")
        self.master.geometry("1000x800")
        self.master.configure(bg='#f0f0f0')

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', font=('Helvetica', 12), background='#f0f0f0')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('Plane.TRadiobutton', font=('Helvetica', 12), padding=10)

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        self.main_frame = ttk.Frame(self.master, padding="10 10 10 10")
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # Option buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        self.classify_button = ttk.Button(self.button_frame, text="Classify Tumor", command=lambda: self.set_mode("classify"))
        self.classify_button.pack(side=tk.LEFT, padx=5)

        self.detect_button = ttk.Button(self.button_frame, text="Detect Tumor", command=lambda: self.set_mode("detect"))
        self.detect_button.pack(side=tk.LEFT, padx=5)

        # Plane selection for detection
        self.plane_frame = ttk.Frame(self.main_frame)
        ttk.Label(self.plane_frame, text="Select Plane:").pack(side=tk.LEFT, padx=5)

        # Load icons
        icon_size = (40, 40)  # Increased icon size
        self.axial_icon = ImageTk.PhotoImage(Image.open(r"P:\Projects\brain tumor danial\Final draft\icons\axial.png").resize(icon_size))
        self.coronal_icon = ImageTk.PhotoImage(Image.open(r"P:\Projects\brain tumor danial\Final draft\icons\coronal.png").resize(icon_size))
        self.sagittal_icon = ImageTk.PhotoImage(Image.open(r"P:\Projects\brain tumor danial\Final draft\icons\sagittal.png").resize(icon_size))

        self.plane_var = tk.StringVar(value="axial")
        ttk.Radiobutton(self.plane_frame, text="Axial", variable=self.plane_var, value="axial", 
                        image=self.axial_icon, compound=tk.LEFT, style='Plane.TRadiobutton').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.plane_frame, text="Coronal", variable=self.plane_var, value="coronal", 
                        image=self.coronal_icon, compound=tk.LEFT, style='Plane.TRadiobutton').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.plane_frame, text="Sagittal", variable=self.plane_var, value="sagittal", 
                        image=self.sagittal_icon, compound=tk.LEFT, style='Plane.TRadiobutton').pack(side=tk.LEFT, padx=5)

        # Image selection and processing
        self.image_frame = ttk.Frame(self.main_frame)
        self.select_button = ttk.Button(self.image_frame, text="Select Image", command=self.select_image)
        self.process_button = ttk.Button(self.image_frame, text="Process", command=self.process_image)
        self.select_button.pack(side=tk.LEFT, padx=5)
        self.process_button.pack(side=tk.LEFT, padx=5)
        self.image_frame.pack(pady=10)

        # Image display
        self.image_label = ttk.Label(self.main_frame, anchor=tk.CENTER)
        self.image_label.pack(expand=True, pady=20)

        # Result display
        self.result_label = ttk.Label(self.main_frame, text="", font=('Helvetica', 14, 'bold'))
        self.result_label.pack(pady=10)

        # Set initial mode
        self.set_mode("classify")

    def set_mode(self, mode):
        self.mode = mode
        if mode == "classify":
            self.plane_frame.pack_forget()
            self.classify_button.state(['disabled'])
            self.detect_button.state(['!disabled'])
        else:
            self.plane_frame.pack(pady=10)
            self.classify_button.state(['!disabled'])
            self.detect_button.state(['disabled'])

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)

    def display_image(self, file_path):
        image = Image.open(file_path)
        image.thumbnail((700, 500))
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def process_image(self):
        if not hasattr(self, 'image_path'):
            messagebox.showerror("Error", "Please select an image first.")
            return

        if self.mode == "classify":
            image = Image.open(self.image_path)
            predicted_class = predict_class(image)
            self.result_label.config(text=f"Predicted Class: {predicted_class}")

        elif self.mode == "detect":
            plane = self.plane_var.get()
            result_img = detect_tumor(self.image_path, plane)
            result_img.thumbnail((700, 500))
            photo = ImageTk.PhotoImage(result_img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.result_label.config(text="Tumor detection complete")

if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()
