import torch
import torchvision.transforms as transforms
from flask import Flask, request, render_template, jsonify
from PIL import Image

# Flask app
app = Flask(__name__)

# Load the ensemble model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ensemble_model = torch.jit.load("super_ensemble_model.pt", map_location=device)
ensemble_model.eval()

# Define image transformations
val_test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define class names
class_names = [
    "Beet Armyworm", "Black Hairy", "Cutworm", "Field Cricket", "Jute Aphid",
    "Jute Hairy", "Jute Red Mite", "Jute Semilooper", "Jute Stem Girdler",
    "Jute Stem Weevil", "Leaf Beetle", "Mealybug", "Pod Borer", "Scopula Emissaria",
    "Termite", "Termite odontotermes (Rambur)", "Yellow Mite"
]

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Load and preprocess the image
        image = Image.open(file).convert("RGB")
        image = val_test_transforms(image)
        image = image.unsqueeze(0).to(device)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            output = ensemble_model(image)
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(output, 1)
            confidence_percentage = confidence.item() * 100

        predicted_label = class_names[predicted_class.item()]
        
        # Return JSON response with prediction and confidence
        return jsonify({
            'prediction': f"Pest Identified: {predicted_label}",
            'confidence': f"{confidence_percentage:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

### Sending the response to the front(html) directly not js file like the above code.

# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from flask import Flask, request, render_template
# from PIL import Image
# import os

# # Flask app
# app = Flask(__name__)

# # Load the ensemble model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ensemble_model = torch.jit.load("ensemble_model.pt", map_location=device)
# ensemble_model.eval()

# # Define image transformations (same as used during training)
# val_test_transforms = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # Define class names (replace with actual class names)
# class_names = ["Beet Armyworm", "Black Hairy", "Cutworm", "Field Cricket", "Jute Aphid", "Jute Hairy", "Jute Red Mite", "Jute Semilooper", "Jute Stem Girdler", "Jute Stem Weevil", "Leaf Beetle", "Mealybug", "Pod Borer", "Scopula Emissaria", "Termite", "Termite odontotermes (Rambur)", "Yellow Mite"]  # Update based on your dataset


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Check if an image is uploaded
#     if 'file' not in request.files:
#         return render_template('index.html', prediction_text="No file uploaded")
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return render_template('index.html', prediction_text="No file selected")

#     try:
#         # Load and preprocess the image
#         image = Image.open(file).convert("RGB")
#         image = val_test_transforms(image)
#         image = image.unsqueeze(0).to(device)  # Add batch dimension

#         # Make prediction
#         with torch.no_grad():
#             output = ensemble_model(image)
#             _, predicted_class = torch.max(output, 1)

#         predicted_label = class_names[predicted_class.item()]

#         return render_template('index.html', prediction_text=f"Predicted Class: {predicted_label}")

#     except Exception as e:
#         return render_template('index.html', prediction_text=f"Error: {str(e)}")


# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000)

# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from flask import Flask, request, jsonify, render_template
# from PIL import Image
# import os

# app = Flask(__name__)

# # Load the saved ensemble model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ensemble_model = torch.jit.load("ensemble_model.pt", map_location=device)
# ensemble_model.eval()

# # Define the same transformations used for validation/testing
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

# # Define class names (same as the dataset classes)
# class_names = [
#     "Beet Armyworm", "Black Hairy", "Cutworm", "Field Cricket", "Jute Aphid", 
#     "Jute Hairy", "Jute Red Mite", "Jute Semilooper", "Jute Stem Girdler", 
#     "Jute Stem Weevil", "Leaf Beetle", "Mealybug", "Pod Borer", 
#     "Scopula Emissaria", "Termite", "Termite odontotermes (Rambur)", "Yellow Mite"
# ]

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('index.html', prediction_text="No file uploaded.")

#     file = request.files['file']
#     if file.filename == '':
#         return render_template('index.html', prediction_text="No file selected.")

#     try:
#         # Load and preprocess the image
#         image = Image.open(file).convert("RGB")
#         image = transform(image).unsqueeze(0).to(device)

#         # Make prediction
#         with torch.no_grad():
#             logits = ensemble_model(image)  # Get raw model outputs (logits)
#             probabilities = F.softmax(logits, dim=1)  # Convert to probabilities
#             predicted_class = torch.argmax(probabilities, dim=1).item()  # Get index of highest probability
#             confidence_percentage = torch.max(probabilities).item() * 100  # Get highest probability as percentage

#         predicted_label = class_names[predicted_class]

#         return render_template(
#             'index.html',
#             prediction_text=f"Prediction: {predicted_label} ({confidence_percentage:.2f}%)"
#         )

#     except Exception as e:
#         return render_template('index.html', prediction_text=f"Error: {str(e)}")

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=5000)
