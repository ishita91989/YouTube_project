import torch
import torch.nn as nn
import torchvision.models as models

# Define the Inception V3 Model structure
class InceptionV3Model(nn.Module):
    def __init__(self):
        super(InceptionV3Model, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 1)  # Adapted for binary classification

    def forward(self, x):
        if self.model.training:
            x, _ = self.model(x)  # Handle auxiliary outputs during training
        else:
            x = self.model(x)
        return torch.sigmoid(x)

# Initialize the global model
global_model = InceptionV3Model()

local_model_paths = ['local_model.pth']  # Example paths
local_models = [InceptionV3Model() for _ in local_model_paths]

# Load the state dicts for each local model
for i, model in enumerate(local_models):
    model.load_state_dict(torch.load(local_model_paths[i]))

# Initialize a state dict to accumulate the average of the local models
global_dict = global_model.state_dict()
for k in global_dict.keys():
    tensors = [local_models[i].state_dict()[k].float() for i in range(len(local_models))]
    global_dict[k] = torch.stack(tensors, 0).mean(0)

# Load the new global model state
global_model.load_state_dict(global_dict)

# Save the updated global model
torch.save(global_model.state_dict(), 'global_model.pth')
print("Global model updated and saved as 'global_model.pth'")
