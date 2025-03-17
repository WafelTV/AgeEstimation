import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) # 3 x 128 x 128 => 64 x 128 x 128 + pooling
        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 64 x 64 x 64 => 128 x 64 x 64 + pooling
        self.conv_bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # 128 x 32 x 32 => 256 x 32 x 32 + final pooling = 256 x 16 x 16
        self.conv_bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_dropout = nn.Dropout2d(0.15)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 16 * 16, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.fc_dropout = nn.Dropout(0.3)
        
        #init.kaiming_uniform_(self.fc1.weight)
        #init.kaiming_uniform_(self.fc2.weight)
        #init.kaiming_uniform_(self.fc3.weight)
        #init.kaiming_uniform_(self.fc4.weight)
    
    def forward(self, x):
        x = self.conv_bn1(nn.functional.leaky_relu(self.pool(self.conv1(x))))
        x = self.conv_bn2(nn.functional.leaky_relu(self.pool(self.conv2(x))))
        x = self.conv_dropout(self.conv_bn3(nn.functional.leaky_relu(self.pool(self.conv3(x)))))
        x = self.flatten(x)
        x = self.fc_dropout(self.bn1(nn.functional.leaky_relu(self.fc1(x))))
        x = self.fc_dropout(self.bn2(nn.functional.leaky_relu(self.fc2(x))))
        x = self.fc_dropout(self.bn3(nn.functional.leaky_relu(self.fc3(x))))
        x = self.fc4(x)
        return x

train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomAutocontrast(),
    transforms.ToTensor()  
])

val_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset_train = ImageFolder("data/age_train_set", transform = train_transforms, target_transform=int)
dataloader_train = DataLoader(dataset_train, shuffle = True, batch_size = 256)

dataset_val = ImageFolder("data/validation_set", transform = val_transforms, target_transform=int)
dataloader_val = DataLoader(dataset_val, shuffle = True, batch_size = 256)

# Device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = CNN_Model().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adamax(network.parameters(), lr=0.001, weight_decay=1e-4)

patience = 10
best_val_loss = float("inf")

for epoch in range(100):
    running_loss = 0.0
    network.train()
    
    #Training Loop
    for images, labels in dataloader_train:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = network(images)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader_train)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    val_loss = 0.0
    network.eval()

    #Validation Loop
    with torch.no_grad():
        for images, labels in dataloader_val:
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item()
    
    val_loss /= len(dataloader_val)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    #Loss Verification
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 10
        torch.save(network.state_dict(), "best_model.pth")
    else:
        patience -= 1
    
    if patience == 0:
        print(f"Training stopped early after {epoch+1} epochs due to no improvement in validation.")
        break
    
network.load_state_dict(torch.load("best_model.pth"))