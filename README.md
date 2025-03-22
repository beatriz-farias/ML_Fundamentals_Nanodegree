# ML Fundamentals Nanodegree Projects Portfolio 🚀

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![SageMaker](https://img.shields.io/badge/Amazon-SageMaker-FF9900?logo=amazonaws)](https://aws.amazon.com/sagemaker/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Comprehensive collection of machine learning projects demonstrating core competencies in neural networks, computer vision, and production workflows. Developed as part of Udacity's Machine Learning Nanodegree program.

## 📚 Table of Contents
- [Project Overview](#-project-overview)
- [Technical Portfolio](#-technical-portfolio)
  - [1. MNIST Handwritten Digits](#1-mnist-handwritten-digits)
  - [2. Bike Sharing Demand](#2-bike-sharing-demand)
  - [3. Landmark Classifier](#3-landmark-classifier)
  - [4. ML Workflow](#4-ml-workflow)
- [🚀 Getting Started](#-getting-started)
- [📊 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [📧 Contact](#-contact)

## 🌐 Project Overview

This repository contains four production-grade machine learning implementations covering fundamental concepts:

| Project | Type | Key Technology | Domain |
|---------|------|-----------------|--------|
| MNIST Classification | Supervised Learning | Feedforward NN | Computer Vision |
| Bike Demand Prediction | Regression Analysis | AutoGluon | Time Series |
| Landmark Recognition | Deep Learning | CNN (ResNet/VGG) | Image Classification |
| ML Workflow Deployment | MLOps | CI/CD Pipelines | Model Monitoring |

## 🛠 Technical Portfolio

### 1. MNIST Handwritten Digits

**📌 Problem Statement**  
Implement a neural network to classify handwritten digits (0-9) from the MNIST dataset.

**🔧 Technical Stack**
- **Framework:** PyTorch/TensorFlow
- **Architecture: **
  - **Type:** Convolutional Neural Network (CNN) for image classification
  - **Layers:** 1 convolutional layer + 3 fully connected (dense) layers
- **Accuracy: **
  - 97.20% on test set, with Adam (LR=0.001)
  - 97.76% on test set, with Adam (LR=0.0005)

**⚙️ Implementation**
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.activation = F.relu
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.fc1 = nn.Linear(32 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        
        return x
```

### 2. Bike Sharing Demand

**📈 Business Objective**
Predict hourly bike rental demand using historical usage patterns and weather data.

#### 🔍 Data Overview
**Dataset**: Capital Bikeshare System (2011-2012)  
**Features**:
- Temporal: `datetime` (parsed to year/month/day/hour features)
- Environmental: `temp`, `atemp`, `humidity`, `windspeed`
- Categorical: `season`, `holiday`, `workingday`, `weather`  
**Target Variable**: `count` (total rentals per hour)

🧠 Model Development
**Automated ML Framework:** AutoGluon TabularPredictor
**Key Configurations:**
```python
predictor = TabularPredictor(
    label="count",
    eval_metric="rmse",
    problem_type="regression"
).fit(
    train_data=train.drop(["casual", "registered"], axis=1),
    time_limit=600,
    presets="best_quality"
)
```
**Training evolution**

Phase	| Features | Hyperparameters | Validation RMSE	| Kaggle Score*
---|---|---|---|---
Initial	| Raw features	| Default	| 51.96	| 1.76244
Enhanced	| + Temporal features	| Default	| 30.06	| 0.70749
Optimized	| Custom feature engineering	| Tuned (LightGBM/XGB/NN)	| 34.97	| 0.43960

*Lower is better

### 3. Landmark Classifier
**🌍 Computer Vision Challenge**
This repository contains a project focused on building a landmark classifier using Convolutional Neural Networks (CNNs). The project explores two primary approaches:

1. 🚀 **Building a CNN from Scratch**: Designing and training a CNN architecture without leveraging pre-trained models.
2. 🧠 **Transfer Learning**: Utilizing pre-trained models to improve classification performance and reduce training time.

#### 📊 Model Comparison: CNN from Scratch vs. Transfer Learning

| Model | Training Time ⏳ | Accuracy 🎯 | Loss 📉 |
|--------|----------------|------------|--------|
| **CNN from Scratch** | ~4 hours | 78% | 0.45 |
| **Transfer Learning (ResNet50)** | ~30 minutes | 92% | 0.21 |

#### 🏗️ Model Architectures

**📌 CNN from Scratch**
The custom CNN model consists of:
- **Convolutional Layers**: 3 convolutional layers with ReLU activation
- **Pooling Layers**: Max pooling layers after each convolution
- **Fully Connected Layers**: Two dense layers with ReLU activation
- **Dropout**: To reduce overfitting
- **Softmax Output Layer**: For multi-class classification

**📌 Transfer Learning (ResNet50)**
The transfer learning model is based on **ResNet50**, a powerful deep learning model pre-trained on ImageNet. The key components include:
- **Pre-trained ResNet50 Base**: Feature extraction using the frozen ResNet50 layers
- **Global Average Pooling**: Reducing spatial dimensions
- **Fully Connected Layer**: Custom dense layer for classification
- **Softmax Output Layer**: Adjusted to match the number of classes

**🔍 Observations:**
- **Transfer Learning significantly improves accuracy (+14%)** while reducing training time by **nearly 8x**. 🚀
- The CNN from scratch model still performs reasonably well, but requires **longer training time** and fine-tuning.
- **Loss is significantly lower** for the transfer learning model, indicating better generalization.

### 4. ML Workflow
**ML Workflow Deployment for Image Classification 🖼️🤖**

End-to-end machine learning workflow for classifying bicycles vs motorcycles from CIFAR-100 images. Features AWS SageMaker deployment, Model Monitoring, and serverless Step Functions orchestration.

#### 📋 Overview
**Business Case**: Automated vehicle classification for logistics optimization at Scones Unlimited  
**Technical Stack**:
- **Data**: CIFAR-100 subset (32x32 RGB images)
- **ML Framework**: SageMaker Image Classification Algorithm
- **Infrastructure**: Lambda + Step Functions + S3
- **Monitoring**: SageMaker Model Monitor


## 🚀 Getting Started
### Prerequisites
- Python 3.8+
- pip >=20.0
- AWS Account with SageMaker access
- S3 Bucket for data storage
- IAM Role with SageMaker/Lambda permissions
- Virtual environment (recommended)

### Installation
```bash
git clone https://github.com/beatriz-farias/ML_Fundamentals_Nanodegree.git
cd ML_Fundamentals_Nanodegree
```
## 📜 License
Distributed under MIT License. See `LICENSE` for more information.
