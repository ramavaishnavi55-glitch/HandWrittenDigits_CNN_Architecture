# HandWrittenDigits_CNN_Architecture

# 🔢 Handwritten Digit Recognition using Convolutional Neural Networks (CNN) with PyTorch

## Project Overview

This project implements a Handwritten Digit Recognition System using Convolutional Neural Networks (CNNs) in PyTorch.

The objective of the project is to automatically recognize handwritten digits from images by learning visual patterns from the MNIST dataset.

Unlike traditional machine learning algorithms that require manual feature engineering, CNNs automatically learn important image features such as edges, curves, shapes, and digit patterns directly from pixel data.

The model was trained using the MNIST handwritten digit dataset and achieved an impressive test accuracy of **98.89%**.

---

## Problem Statement

Handwritten digit recognition is a fundamental problem in computer vision and deep learning.

Applications include:

* Postal code recognition
* Bank cheque processing
* Form digitization
* Automated document analysis
* OCR (Optical Character Recognition) systems

The objective of this project is to:

* Build a CNN model using PyTorch
* Train the model on handwritten digit images
* Classify digits from 0 to 9
* Evaluate model performance on unseen test data

---

## Project Workflow

MNIST Dataset Collection

↓

Data Preprocessing

↓

Dataset Exploration

↓

Create DataLoaders

↓

Build CNN Architecture

↓

Define Loss Function

↓

Configure Optimizer

↓

Train CNN Model

↓

Evaluate Model Performance

↓

Visualize Predictions

↓

Save Trained Model

---

## Dataset Information

The project uses the MNIST dataset, one of the most popular datasets for image classification.

### Dataset Characteristics

| Property          | Value          |
| ----------------- | -------------- |
| Training Images   | 60,000         |
| Testing Images    | 10,000         |
| Image Size        | 28 × 28 Pixels |
| Channels          | 1 (Grayscale)  |
| Number of Classes | 10             |
| Labels            | 0 – 9          |

### Output

Training Images: 60000

Testing Images: 10000

Image Shape: torch.Size([1, 28, 28])

Label: 5

---

## Data Preprocessing

The following preprocessing step was applied:

### Tensor Conversion

The images were converted from PIL image format into PyTorch tensors using:

```python
transforms.ToTensor()
```

### Why Tensor Conversion?

Deep learning models operate on tensors rather than image objects. Tensor conversion transforms image pixels into numerical values that can be processed by neural networks.

---

## Dataset Exploration

Before training the model, the dataset was explored to understand image structure and pixel distributions.

### Image Shape

```text
torch.Size([1, 28, 28])
```

#### Meaning

* 1 Channel (Grayscale)
* Height = 28
* Width = 28

---

### Pixel Value Range

Output:

```text
Minimum Pixel Value: tensor(0.)
Maximum Pixel Value: tensor(1.)
```

#### Observation

The images were automatically normalized between 0 and 1 by the ToTensor() transformation.

---

## DataLoader Creation

Mini-batch training was implemented using PyTorch DataLoaders.

### Configuration

Batch Size = 64

### Output

```text
Images Shape: torch.Size([64, 1, 28, 28])

Labels Shape: torch.Size([64])
```

### Interpretation

Each training batch contains:

* 64 Images
* 1 Channel
* 28 × 28 Pixels
* 64 Corresponding Labels

---

## CNN Architecture

A Convolutional Neural Network was designed for digit classification.

### Model Structure

#### Convolution Layer 1

```python
Conv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=3,
    padding=1
)
```

Purpose:

Extracts low-level image features such as edges and basic shapes.

---

#### Convolution Layer 2

```python
Conv2d(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    padding=1
)
```

Purpose:

Learns more complex patterns such as digit structures and curves.

---

#### Max Pooling Layer

```python
MaxPool2d(2,2)
```

Purpose:

Reduces image dimensions while preserving important features.

---

#### Fully Connected Layer 1

```python
Linear(3136,128)
```

Purpose:

Combines extracted features into a compact representation.

---

#### Fully Connected Layer 2

```python
Linear(128,10)
```

Purpose:

Produces probabilities for digits 0–9.

---

### Complete Model Output

```text
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2)
  (fc1): Linear(in_features=3136, out_features=128)
  (fc2): Linear(in_features=128, out_features=10)
)
```

---

## Output Shape Verification

Output:

```text
torch.Size([64, 10])
```

### Interpretation

For every batch of 64 images, the network outputs probabilities for 10 digit classes.

---

## Loss Function

### CrossEntropyLoss

```python
CrossEntropyLoss()
```

### Definition

CrossEntropyLoss measures the difference between the predicted class probabilities and the actual class labels.

### Output

```text
CrossEntropyLoss()
```

---

## Optimizer

### Adam Optimizer

```python
Adam(
    lr = 0.001
)
```

### Definition

Adam is an adaptive optimization algorithm that updates model weights efficiently during training.

### Output

```text
Adam
(
Parameter Group 0
lr: 0.001
)
```

---

## Model Training

The CNN model was trained for:

```text
Epochs = 5
```

### Training Results

```text
Epoch [1/5], Loss: 0.1723

Epoch [2/5], Loss: 0.0492

Epoch [3/5], Loss: 0.0343

Epoch [4/5], Loss: 0.0242

Epoch [5/5], Loss: 0.0191
```

### Observation

The loss consistently decreased across epochs, indicating successful learning and convergence.

---

## Model Evaluation

After training, the model was evaluated on the test dataset.

### Output

```text
Test Accuracy: 98.89%
```

### Interpretation

The model correctly classified approximately:

```text
98.89 out of every 100 handwritten digits
```

This demonstrates excellent generalization performance.

---

## Prediction Visualization

The trained CNN was used to predict handwritten digits from the test dataset.

### Procedure

1. Select images from test dataset.
2. Pass images through the trained CNN.
3. Obtain predicted labels.
4. Compare predicted labels with actual labels.
5. Display results visually.

### Observation

Most predictions matched the true digit labels, confirming the model's high classification accuracy.

---

## Concepts Learned

### Tensor

A tensor is a multidimensional numerical array used as the primary data structure in PyTorch.

### MNIST Dataset

A benchmark dataset containing handwritten digit images used for image classification tasks.

### DataLoader

A utility that loads data in mini-batches during training.

### Batch Size

The number of samples processed before updating model weights.

### Convolution Layer

A layer that automatically extracts important visual features from images.

### Feature Maps

Intermediate outputs generated by convolution layers that represent learned image patterns.

### Max Pooling

A downsampling operation that reduces image dimensions while retaining important information.

### Fully Connected Layer

A layer that combines extracted features and performs final classification.

### ReLU

An activation function that introduces non-linearity into neural networks.

### CrossEntropyLoss

A loss function used for multi-class classification problems.

### Adam Optimizer

An optimization algorithm that adjusts weights using adaptive learning rates.

### Epoch

One complete pass through the entire training dataset.

### Forward Propagation

The process of passing input data through the network to generate predictions.

### Backpropagation

The process of updating network weights based on prediction errors.

### CNN

A deep learning architecture specifically designed for image processing tasks.

### Image Classification

The task of assigning images to predefined categories.

---

## Skills Demonstrated

* Deep Learning
* Computer Vision
* PyTorch
* CNN Architecture Design
* Image Classification
* Data Preprocessing
* Neural Networks
* Model Evaluation
* Prediction Visualization
* Deep Learning Workflow

---

## Repository Structure

Handwritten-Digit-Recognition-CNN/

├── digit_recognition.py

├── saved_model.pth

├── README.md

└── requirements.txt

---

## Future Improvements

* Train for additional epochs.
* Add dropout layers to reduce overfitting.
* Implement data augmentation.
* Build a web application using Streamlit.
* Allow users to upload custom handwritten digits.
* Deploy the model online.

---

## Conclusion

This project successfully demonstrates the implementation of a Convolutional Neural Network using PyTorch for handwritten digit recognition. The CNN automatically learned visual features from image pixels and achieved a test accuracy of **98.89%** on the MNIST dataset.

The project provided hands-on experience in deep learning, computer vision, neural network design, model training, and evaluation. The final model successfully classifies handwritten digits and serves as a strong foundation for more advanced deep learning and image recognition projects.

