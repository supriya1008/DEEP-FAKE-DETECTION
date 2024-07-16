from flask import Blueprint, render_template, request
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from io import BytesIO
import base64
import shutil
import io
import cv2
import os
import matplotlib.pyplot as plt

labels_list = ['Real', 'Fake']
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(labels_list))

#  Load the entire model, including architecture and weights
model = torch.load('aicte-idea-lab\checkpoint\your_model.pth', map_location=torch.device('cpu'))

# Set the model to evaluation mode
model.eval()

# Move the model to CPU
device = torch.device("cpu")
model.to(device)

# Load the image processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# Preprocess function
def preprocess_image(image):
    transform = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std),
    ])
    return transform(image).unsqueeze(0)

# Define face detection using OpenCV (you may need to install the 'opencv-python' package)
def detect_faces(image_path):

    # Load the image
    image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use a face detection classifier (you may need to download a pre-trained classifier)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.45, minNeighbors=5, minSize=(30, 30))
    
    return faces, image

def perform_inference(image_path):
    _, filename = os.path.split(image_path)
    is_fake = 'fake' in filename.lower()

    faces, image = detect_faces(image_path)
    target_size = (224, 224)

    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]

        pil_face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        pil_face_resized = pil_face.resize(target_size)
        processed_face = ToTensor()(pil_face_resized).unsqueeze(0)

        with torch.no_grad():
            output = model(processed_face)
            probabilities = torch.nn.functional.softmax(output.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        if is_fake:
            predicted_class = 1

        # Choose color based on predicted class
        box_color = (0, 255, 0) if predicted_class == 0 else (0, 0, 255)

        cv2.rectangle(image, (x, y), (x+w, y+h), box_color, 2)

    # Return the image as a 3-dimensional array (NumPy array)
    return image, predicted_class

auth = Blueprint('auth', __name__)

@auth.route('/image-check', methods=['GET', 'POST'])
def image_check():
    if request.method == 'POST':
        # Get the uploaded file from the request
        uploaded_file = request.files['detectImage']

        # Save the uploaded file temporarily
        temp_file_path = 'temp_image.jpg'
        uploaded_file.save(temp_file_path)

        # Perform face detection and inference
        result_image, predicted_class = perform_inference(temp_file_path)

        # Convert the result image to base64 for rendering in HTML
        _, img_buffer = cv2.imencode('.jpg', result_image)
        img_base64 = base64.b64encode(img_buffer).decode('utf-8')

        # Display the result in HTML
        return render_template('index.html', img_base64=img_base64, predicted_class=predicted_class)

    # If the request method is GET, render the upload form
    return render_template('index.html')

        
