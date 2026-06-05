import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim


# Transform images to tensors
transform = transforms.ToTensor()

# Download training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Download testing dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("Training Images:", len(train_dataset))
print("Testing Images:", len(test_dataset))

image, label = train_dataset[0]

print("Image Shape:", image.shape)
print("Label:", label)



# Display first image
image, label = train_dataset[0]

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()




fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i in range(10):
    image, label = train_dataset[i]

    axes[i // 5, i % 5].imshow(image.squeeze(), cmap='gray')
    axes[i // 5, i % 5].set_title(f"{label}")
    axes[i // 5, i % 5].axis('off')

plt.tight_layout()
plt.show()

image, label = train_dataset[0]

print("Minimum Pixel Value:", image.min())
print("Maximum Pixel Value:", image.max())





# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

images, labels = next(iter(train_loader))

print("Images Shape:", images.shape)
print("Labels Shape:", labels.shape)


images, labels = next(iter(train_loader))

plt.imshow(images[0].squeeze(), cmap='gray')
plt.title(f"Label: {labels[0].item()}")
plt.axis('off')
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # First Convolution Layer
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # Second Convolution Layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        # Max Pooling Layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x
    
model = CNN()

print(model)    


sample_images, _ = next(iter(train_loader))

output = model(sample_images)

print(output.shape)
criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print(criterion)
print(optimizer)

num_epochs = 5

for epoch in range(num_epochs):

    running_loss = 0.0

    for images, labels in train_loader:

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(images)

        # Calculate Loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}], "
        f"Loss: {running_loss/len(train_loader):.4f}"
    )


correct = 0
total = 0

# Disable gradient calculations
with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        # Predicted class
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")    


"""
import matplotlib.pyplot as plt

# Get first test image
image, label = test_dataset[0]

# Add batch dimension
image_batch = image.unsqueeze(0)

# Prediction mode
model.eval()

with torch.no_grad():
    output = model(image_batch)
    _, prediction = torch.max(output, 1)

print("Actual Label:", label)
print("Predicted Label:", prediction.item())

plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Actual: {label} | Predicted: {prediction.item()}")
plt.axis('off')
plt.show()
"""

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

model.eval()

for i in range(10):

    image, label = test_dataset[i]

    image_batch = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image_batch)
        _, prediction = torch.max(output, 1)

    axes[i//5, i%5].imshow(
        image.squeeze(),
        cmap='gray'
    )

    axes[i//5, i%5].set_title(
        f"P:{prediction.item()} A:{label}"
    )

    axes[i//5, i%5].axis('off')

plt.tight_layout()
plt.show()
"""Output:
Training Images: 60000
Testing Images: 10000
Image Shape: torch.Size([1, 28, 28])
Label: 5
Minimum Pixel Value: tensor(0.)
Maximum Pixel Value: tensor(1.)
Images Shape: torch.Size([64, 1, 28, 28])
Labels Shape: torch.Size([64])
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=3136, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
torch.Size([64, 10])
CrossEntropyLoss()
Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    decoupled_weight_decay: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.001
    maximize: False
    weight_decay: 0
)
Epoch [1/5], Loss: 0.1723
Epoch [2/5], Loss: 0.0492
Epoch [3/5], Loss: 0.0343
Epoch [4/5], Loss: 0.0242
Epoch [5/5], Loss: 0.0191
Test Accuracy: 98.89%"""
