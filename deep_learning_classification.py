# -*- coding: utf-8 -*-
"""Deep_Learning_HW1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17Owc6b6rgqYtgoOrGu89M24vqAApir26
"""

!pip install kaggle

import kagglehub

# Download latest version
path = kagglehub.dataset_download("gpiosenka/sports-classification")

print("Path to dataset files:", path)

import shutil
import kagglehub

# Download to default location
path = kagglehub.dataset_download("gpiosenka/sports-classification")

# Move to custom path
custom_path = "/content/drive/MyDrive/sports"
shutil.move(path, custom_path)

print("Path to dataset files:", custom_path)

"""# Data Loading"""

import torchvision
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for RGB channels
])

dataset_path = '/content/drive/MyDrive/sports'  # Adjust path to your dataset
train_data = datasets.ImageFolder(root=dataset_path+'/train', transform=transform)

# Optionally, use DataLoader to handle batching and shuffling
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


dataset_path = '/content/drive/MyDrive/sports'  # Adjust path to your dataset
test_data = datasets.ImageFolder(root=dataset_path+'/test', transform=transform)

# Optionally, use DataLoader to handle batching and shuffling
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

print(len(train_data.classes))

"""# Data Visualisation"""

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# Function to show images
def show_images(loader):
    # Get a batch of data
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # Make a grid from the batch of images
    img_grid = make_grid(images, nrow=8, padding=2, normalize=True)

    # Convert to numpy for plotting
    np_img = img_grid.numpy().transpose((1, 2, 0))

    # Display the images
    plt.figure(figsize=(12, 12))
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()

# Call the function to display images from valid_loader
show_images(train_loader)

!pip install timm
import timm
model_names = timm.list_models(pretrained=True)
model_names

"""# 1. Model resnext26ts.ra2_in1k"""

model = timm.create_model('resnext26ts.ra2_in1k', pretrained=True)

print(f'Original pooling: {model.global_pool}')
print(f'Original classifier: {model.get_classifier()}')
print('--------------------')

model.reset_classifier(100, 'max')
print(f'Modified pooling: {model.global_pool}')
print(f'Modified classifier: {model.get_classifier()}')

model.default_cfg

"""## Model train and test"""

num_in_features = model.get_classifier().in_features
model.fc = nn.Sequential(
    nn.BatchNorm1d(num_in_features),
    nn.Linear(in_features=num_in_features, out_features=512, bias=False),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(in_features=512, out_features=100, bias=False))

from torch import nn
# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    device = torch.device('cuda')  # CUDA GPU
elif torch.backends.mps.is_available():
    device = torch.device('mps') #Apple GPU
else:
    device = torch.device("cpu")

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Start the timer
from timeit import default_timer as timer
start_time = timer()

# Setup training and save the results
results = train(model=model,
                       train_dataloader=train_loader,
                       test_dataloader=test_loader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=5,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

"""## Accuracy and Loss Curves"""

import matplotlib.pyplot as plt
# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

plot_loss_curves(results)

"""## Test on custom image"""

from typing import List, Tuple

from PIL import Image

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):


    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ###

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)

    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);
    return class_names[target_image_pred_label]

# Make predictions on and plot the images
image_path = "/content/drive/MyDrive/sports/test/baseball/1.jpg"
classes = train_data.classes
pred_and_plot_image(model=model,
                        image_path=image_path,
                        class_names=classes,
                        # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                        image_size=(224, 224))

"""## Model export"""

!pip install onnx

import torch
import torch.onnx

# Assuming m is your pre-trained or custom model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move your model to the device

# Export the model to ONNX
image_height = 224
image_width = 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True).to(device)  # Move input tensor to the same device
torch_out = model(x)

torch.onnx.export(model,                     # model being run
                  x,                     # model input (or a tuple for multiple inputs)
                  "model_v1.onnx",       # where to save the model (can be a file or file-like object)
                  export_params=True,    # store the trained parameter weights inside the model file
                  opset_version=12,      # the ONNX version to export the model to
                  do_constant_folding=True, # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output']) # the model's output names

print("Model exported to model_v1.onnx")

"""# 2. Model 'resnext26ts.ra2_in1k' with modifications"""



model1 = timm.create_model('resnext26ts.ra2_in1k', pretrained=True)

print(f'Original pooling: {model1.global_pool}')
print(f'Original classifier: {model1.get_classifier()}')
print('--------------------')

model1.reset_classifier(100, 'max')

print(f'Modified pooling: {model1.global_pool}')
print(f'Modified classifier: {model1.get_classifier()}')

"""## Model train and test"""

num_in_features = model1.get_classifier().in_features
model1.fc = nn.Sequential(
    nn.BatchNorm1d(num_in_features),
    nn.Linear(in_features=num_in_features, out_features=512, bias=False),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(in_features=512, out_features=100, bias=False))

from torch import nn
import torch.optim as optim

# Define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

from torch.cuda.amp import autocast
from typing import Tuple
import torch
from tqdm import tqdm

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:

    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass with autocast for mixed precision
        with autocast():
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        train_loss += loss.item()

        # 2. Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 3. Step the optimizer
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          epochs: int,
          device: torch.device) -> dict:

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        scheduler.step()  # Step the scheduler at the end of each epoch if needed

    return results

# Start the training
results = train(model=model1,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                epochs=5,
                device=device)

print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

"""## Accuracy and Loss curves"""

import matplotlib.pyplot as plt
# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

plot_loss_curves(results)

"""## Test on custom image"""

from typing import List, Tuple

from PIL import Image

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):


    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ###

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)

    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);
    return class_names[target_image_pred_label]

# Make predictions on and plot the images
image_path = "/content/drive/MyDrive/sports/test/baseball/1.jpg"
classes = train_data.classes
pred_and_plot_image(model=model,
                        image_path=image_path,
                        class_names=classes,
                        # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                        image_size=(224, 224))

"""## Model export"""

!pip install onnx

import torch
import torch.onnx

# Assuming m is your pre-trained or custom model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move your model to the device

# Export the model to ONNX
image_height = 224
image_width = 224
x = torch.randn(1, 3, image_height, image_width, requires_grad=True).to(device)  # Move input tensor to the same device
torch_out = model(x)


torch.onnx.export(model,                     # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "model_v2.onnx",           # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # ONNX version
                  do_constant_folding=True,  # optimization
                  input_names=['input'],     # input names
                  output_names=['output'],   # output names
                  dynamic_axes={
                      'input': {0: 'batch_size'},  # dynamic batch size for input
                      'output': {0: 'batch_size'}  # dynamic batch size for output
                  })

print("Model exported to model_v2.onnx.")

"""## Inference Optimization"""

import onnxruntime as ort
import numpy as np

# Load the ONNX model
session = ort.InferenceSession("model_v1.onnx")

# Prepare input (the shape should match the model's expected input)
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)  # Example input shape

# Run inference
outputs = session.run(None, {input_name: input_data})
print(outputs)

import time

# Measure inference time before optimization
start_time = time.time()
outputs = session.run(None, {input_name: input_data})
end_time = time.time()
print(f"Inference time (without optimization): {end_time - start_time} seconds")

# If you've applied optimizations (e.g., graph optimization, quantization, or GPU), rerun the timing:
start_time = time.time()
outputs = session.run(None, {input_name: input_data})
end_time = time.time()
print(f"Inference time (after optimization): {end_time - start_time} seconds")



import torch
import time
import matplotlib.pyplot as plt

def evaluate_latency_and_accuracy(model1, dataloader, device):
    """Evaluates latency and accuracy of the model."""
    model1.eval()  # Set to evaluation mode
    model1.to(device)

    total_correct = 0
    total_samples = 0
    latencies = []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Measure inference time
            start_time = time.time()
            y_pred = model(X)
            end_time = time.time()

            # Calculate latency
            latency = end_time - start_time
            latencies.append(latency)

            # Calculate accuracy
            y_pred_labels = torch.argmax(y_pred, dim=1)
            total_correct += (y_pred_labels == y).sum().item()
            total_samples += y.size(0)

    # Calculate average latency
    avg_latency = sum(latencies) / len(latencies)
    accuracy = total_correct / total_samples

    return accuracy, avg_latency

# Convert model to TorchScript for inference optimization
optimized_model = torch.jit.script(model1)

# Evaluate both models
baseline_accuracy, baseline_latency = evaluate_latency_and_accuracy(model1, test_loader, device)
optimized_accuracy, optimized_latency = evaluate_latency_and_accuracy(optimized_model, test_loader, device)

# Plot the comparison graph
labels = ['Baseline Model', 'Optimized Model']
accuracies = [baseline_accuracy, optimized_accuracy]
latencies = [baseline_latency, optimized_latency]

fig, ax1 = plt.subplots()

# Plot accuracy
ax1.set_xlabel('Model')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.bar(labels, accuracies, color='tab:blue', alpha=0.6, label='Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot latency on a second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Latency (seconds)', color='tab:red')
ax2.plot(labels, latencies, color='tab:red', marker='o', linestyle='--', label='Latency')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Title and legend
plt.title('Model Accuracy and Latency Comparison')
fig.tight_layout()
plt.show()

