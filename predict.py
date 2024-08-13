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
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.train()

num_traits = 6
traits =  np.empty((0, 0))
img_dir = './data/train_images'
img_test_dir = './data/test_images'
train_csv_dir = './data/train.csv'
test_csv_dir = './data/test.csv'
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

class PlantImgTrainDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
      # read the last 6 columns => traits
        df_header = pd.read_csv(train_csv_dir, nrows=0)
        # normalizing traits
        self.plant_img_traits = pd.read_csv(train_csv_dir, usecols=df_header.columns[-6:])
        self.min_train = np.min(self.plant_img_traits, axis=0)
        min_train = self.min_train
        self.max_train = np.max(self.plant_img_traits, axis=0)
        max_train = self.max_train
        f = open('max_train.pckl', 'wb')
        pickle.dump(max_train, f)
        f.close()
        f = open('min_train.pckl', 'wb')
        pickle.dump(min_train, f)
        f.close()
        self.plant_img_traits = (self.plant_img_traits - self.min_train) / (self.max_train - self.min_train)
        self.plant_img_id = pd.read_csv(train_csv_dir, usecols=[0])

        self.plant_img_dir = img_dir

        self.transform = transforms.Compose([
                              transforms.Resize((224, 224)),  # Resize the image to 224x224
                               transforms.ToTensor()
                          ])


    def __len__(self):
        return len(self.plant_img_traits)

    def __getitem__(self, idx):
        id_plant_img = np.array(self.plant_img_id).astype(str)
        img_path = os.path.join(self.plant_img_dir, id_plant_img[idx,0] + '.jpeg')
        image = Image.open(img_path)
        image = self.transform(image)
        return image, torch.tensor(self.plant_img_traits.values[idx], dtype=torch.float32)

class PlantImgTestDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.plant_img_test_dir = img_test_dir
        self.plant_test_img_id = pd.read_csv(test_csv_dir, usecols=[0])
        self.transform = transforms.Compose([
                              transforms.Resize((224, 224)),  # Resize the image to 224x224
                               transforms.ToTensor()
                          ])

    def __len__(self):
        return len(self.plant_test_img_id)

    def __getitem__(self, idx):
        id_plant_test_img = np.array(self.plant_test_img_id).astype(str)
        img_path = os.path.join(self.plant_img_test_dir, id_plant_test_img[idx, 0] + '.jpeg')
        image = Image.open(img_path)
        image = self.transform(image)
        return image

def train_model(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        progress = 0
        running_loss = 0.0
        for images, traits in dataloader:
            images, traits = images.to(device), traits.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, traits)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress = progress + 1
            if progress%10 == 0:
                print(f'progress:',progress)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}')

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor()
])

train_data = PlantImgTrainDataset()
test_data = PlantImgTestDataset()
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model_plant_trait_prediction = PlantTraitPredictionCNN(num_traits).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

output_csv_path = './data/predictions_resnet152_final.csv'

model_loaded = PlantTraitPredictionCNN(num_traits)
model_loaded.to(device)
model_saved= torch.load('./trained_model_batch32_2_epoch_resnet152.pth')
model_loaded.load_state_dict(model_saved)

f = open('max_train.pckl', 'rb')
max_train = pickle.load(f)
f.close()
f = open('min_train.pckl', 'rb')
min_train = pickle.load(f)
f.close()
max_train = max_train.values.reshape(1, -1)
min_train = min_train.values.reshape(1, -1)
# max_train = max_train.reshape(1, -1)
# min_train = min_train.reshape(1, -1)
predict_and_save(model_loaded, testloader, output_csv_path)