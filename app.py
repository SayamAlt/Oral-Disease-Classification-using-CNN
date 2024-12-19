from flask import Flask, request, render_template, jsonify
import torch, warnings
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")
from PIL import Image
from torchvision import transforms

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

app = Flask(__name__)
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) # 64,64,32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # 32,32,64
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 16,16,128
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1) # 8,8,256
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(in_features=8*8*256, out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=1)

    def forward(self, X):
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.bn2(self.conv2(X)))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.bn3(self.conv3(X)))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.bn4(self.conv4(X)))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = X.view(-1,8*8*256)
        X = F.relu(self.fc1(X))
        X = F.dropout(X,p=0.2)
        X = F.relu(self.fc2(X))
        X = F.dropout(X,p=0.1)
        X = F.relu(self.fc3(X))
        return self.fc4(X)  # Raw output for use with BCEWithLogitsLoss
    
# Load the trained model
model = torch.load('oral_disease_classifier.pt',map_location=device)

# Define image preprocessing transformations
img_transforms = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.CenterCrop((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize image to match model's input requirements
])

model.eval() # Set model to evaluation mode

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