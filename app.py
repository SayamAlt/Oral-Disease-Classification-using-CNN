from flask import Flask, request, render_template, jsonify
import torch, warnings
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")
from PIL import Image
from torchvision import transforms
from models import CNN
    
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

app = Flask(__name__)
    
# Load the trained model
model = CNN()
model.load_state_dict(torch.load('oral_disease_classifier_state_dict.pt', map_location=device))
model = model.to(device)
model.eval() # Set model to evaluation mode

# Define image preprocessing transformations
img_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.CenterCrop((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize image to match model's input requirements
])

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        image_file = request.files['image']

        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        try:
            # Open and preprocess the image file
            image = Image.open(image_file).convert("RGB")
            transformed_img = img_transforms(image).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(transformed_img).item()

            if int(pred >= 0.5) == 1:
                return render_template('index.html',prediction_text="The predicted oral disease is gingivitis.")
            
            return render_template('index.html',prediction_text="The predicted oral disease is caries.")
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
if __name__ == "__main__":
    app.run(port=8000,debug=True)