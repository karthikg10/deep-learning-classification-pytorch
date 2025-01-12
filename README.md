# deep-learning-classification-pytorch

# Deep Learning Project: Sports Classification

This project focuses on building a deep learning model for multi-class classification of sports images using the PyTorch framework. The dataset for this project is sourced from Kaggle and consists of images categorized by different sports.

---

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training and Testing](#training-and-testing)
- [Model Export](#model-export)
- [Inference Optimization](#inference-optimization)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Dataset

The dataset is downloaded from Kaggle using the `kagglehub` library. It contains images categorized by sport and is divided into `train` and `test` subsets.

Path to Dataset: `/content/drive/MyDrive/sports`

---

## Model Architecture

The project uses the `resnext26ts.ra2_in1k` model from the `timm` library. Modifications include:

- Changing the classifier to adapt to 100 classes.
- Adding BatchNorm, ReLU activation, and Dropout layers.

---

## Data Preprocessing

- Images are resized to `224x224`.
- Normalized using mean `[0.5, 0.5, 0.5]` and std `[0.5, 0.5, 0.5]`.
- PyTorch `DataLoader` is used for batching and shuffling.

---

## Training and Testing

### Training:
- Loss Function: CrossEntropyLoss
- Optimizer: Adam (for the first model), SGD with momentum (for the second model)
- Training runs for 5 epochs with GPU or CPU support.

### Testing:
- Model performance is evaluated on the test set using accuracy and loss metrics.

---

## Model Export

The trained model is exported to the ONNX format for interoperability with other frameworks and optimization tools:
- Filename: `model_v1.onnx` and `model_v2.onnx`

---

## Inference Optimization

- Used ONNX Runtime for faster inference.
- Evaluated latency and accuracy for the baseline and optimized models.
- Optimized the PyTorch model using TorchScript.

---

## Results

### Model Performance:
- Plotted accuracy and loss curves for training and testing phases.
- Evaluated latency and accuracy trade-offs for the baseline and optimized models.

---

## How to Run

1. Install required dependencies:
   ```bash
   pip install kagglehub timm torch torchvision onnx onnxruntime
   
Download the dataset using KaggleHub.

Run the script to train, test, and export the model:

python deep_learning_classification.py

To test the model on custom images: Place the image in the appropriate directory and use the pred_and_plot_image function.
