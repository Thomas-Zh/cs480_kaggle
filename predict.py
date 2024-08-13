import torch
import torch.nn as nn
import os
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.optim as optim
import pickle
num_traits = 6
traits =  np.empty((0, 0))
img_dir = '/Users/xiaofeizhang/cs680/kaggle/data/train_images'
img_test_dir = '/Users/xiaofeizhang/cs680/kaggle/data/test_images'
train_csv_dir = '/Users/xiaofeizhang/cs680/kaggle/data/train.csv'
test_csv_dir = '/Users/xiaofeizhang/cs680/kaggle/data/test.csv'
min_train = []
min_train = np.array(min_train)
max_train = []
max_train = np.array(max_train)


class PlantTraitPredictionCNN(nn.Module):
    def __init__(self, num_traits):
        super(PlantTraitPredictionCNN, self).__init__()
        self.resnet = model
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_traits)

    def forward(self, x):
       return self.resnet(x)
def predict_and_save(model, dataloader, csv_path):
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for images in dataloader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()  # Get predictions and move to CPU
            predictions.append(outputs)
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    
    # Un-normalize the predictions
    unnormalized_predictions = predictions * (max_train - min_train) + min_train
    
    # Get the IDs of the test images
    test_img_ids = dataloader.dataset.plant_test_img_id
    
    # Create a DataFrame to hold the predictions
    prediction_df = pd.DataFrame(unnormalized_predictions, columns=["X4", "X11", "X18", "X26", "X50", "X3112"])
    prediction_df.insert(0, "id", test_img_ids)
    
    # Save the predictions to a CSV file
    prediction_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

output_csv_path = '/Users/xiaofeizhang/cs680/kaggle/data/predictions_resnet152_final.csv'
model_loaded = PlantTraitPredictionCNN(num_traits)
model_loaded.to(device)
model_saved= torch.load('/Users/xiaofeizhang/cs680/kaggle/trained_model_batch32_2_epoch_resnet152.pth')
model_loaded.load_state_dict(model_saved)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
f = open('max_train.pckl', 'rb')
max_train = pickle.load(f)
f.close()
f = open('min_train.pckl', 'rb')
min_train = pickle.load(f)
f.close()
max_train = max_train.values.reshape(1, -1)
min_train = min_train.values.reshape(1, -1)

predict_and_save(model_loaded, testloader, output_csv_path)