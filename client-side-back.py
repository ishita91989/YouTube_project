import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define the model structure using Inception V3
class InceptionV3Model(nn.Module):
    def __init__(self):
        super(InceptionV3Model, self).__init__()
        # Load pre-trained Inception V3 model
        inception_v3 = models.inception_v3(pretrained=True)
        
        # Freeze all the parameters in the feature extraction layers
        for param in inception_v3.parameters():
            param.requires_grad = False

        # Replace the classifier part of Inception V3
        num_features = inception_v3.fc.in_features
        inception_v3.fc = nn.Linear(num_features, 1)  # Binary classification

        self.model = inception_v3

    def forward(self, x):
        # Inception V3's forward method may return auxiliary outputs
        # when training, which we do not need during inference
        if self.model.training:
            x, _ = self.model(x)
        else:
            x = self.model(x)
        return x

# Load the global model
local_model = InceptionV3Model()
local_model.load_state_dict(torch.load('global_model.pth'))

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Update size for Inception V3
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for ImageNet
])

# Load the local dataset
train_data = datasets.ImageFolder('screenshots/', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Define the loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss
optimizer = optim.Adam(local_model.parameters(), lr=0.001)

# Train the model locally
local_model.train()
for epoch in range(5):  # Local training for 5 epochs
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = local_model(images)
        loss = criterion(outputs, labels.float().unsqueeze(1))
        loss.backward()
        optimizer.step()

# Save the updated local model
torch.save(local_model.state_dict(), 'local_model.pth')

print("Local model trained and saved as 'local_model.pth'")