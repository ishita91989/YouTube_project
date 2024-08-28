from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms, models
from PIL import Image
import io
import base64
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "chrome-extension://icmkkikocohmeeadkacldigogkddidfb"}})

logging.basicConfig(level=logging.DEBUG)

class InceptionV3Model(torch.nn.Module):
    def __init__(self):
        super(InceptionV3Model, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        return self.model(x)

model = InceptionV3Model()
model.load_state_dict(torch.load('global_model.pth'))
model.eval()

def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

@app.route('/classify', methods=['POST'])
def classify_image():
    logging.debug("Received request with data: %s", request.data)
    data = request.get_json(force=True)
    image_data = data.get('image')
    if not image_data:
        logging.error("No image data found in request")
        return jsonify({'error': 'No image data'}), 400
    
    image_bytes = base64.b64decode(image_data.split(',')[1])
    tensor = transform_image(image_bytes)
    output = model(tensor)
    prediction = torch.sigmoid(output).item() > 0.5
    logging.debug("Prediction: %s", prediction)
    return jsonify({'classification': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
