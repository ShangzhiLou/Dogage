# Dog Age Classification Project

This project implements deep learning models to classify dog images into three age categories: Young, Adult, and Senior.

## Project Overview

The goal of this project is to develop accurate deep learning models for dog age classification using transfer learning with pre-trained CNN architectures. The project includes:

- Data loading and preprocessing
- Model training and evaluation
- Results visualization
- Multiple model comparison

## Dataset

The dataset consists of dog images from two sources:
1. Expert_Train dataset
2. PetFinder_All dataset

Images are organized into three age categories:
- Young
- Adult 
- Senior

## Model Architectures

The project implements and compares three CNN architectures:

1. MobileNetV2
2. EfficientNetB0
3. ResNet18

All models are pre-trained on ImageNet and fine-tuned for the dog age classification task.

## Training Process

Key training parameters:
- Image size: 224x224
- Batch size: 32
- Epochs: 20
- Optimizer: Adam
- Learning rate: 0.001
- Data augmentation: Horizontal flip, rotation, color jitter
- Face detection preprocessing

Training logs are saved for each model, including:
- Training/validation loss and accuracy
- Learning rate schedule
- Best model weights

## Evaluation

Models are evaluated on a held-out test set with:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix

## Visualization

The project includes visualization capabilities through `visualize_results.py` which generates:
- Training curves (loss and accuracy)
- Classification report heatmaps
- Confusion matrices

## Environment Setup

The project was developed and tested on the following environment:

- **Hardware**: NVIDIA RTX 4090 GPU (AutoDL cloud server)
- **Software**:
  - Miniconda Python environment
  - CUDA 11.7
  - cuDNN 8.5
- **Python Packages**:
  - Python 3.7+
  - PyTorch with GPU support
  - Albumentations
  - Pandas
  - Matplotlib
  - Seaborn
  - OpenCV
  - tqdm

To set up the environment:

```bash
conda create -n dogage python=3.8
conda activate dogage
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install albumentations pandas matplotlib seaborn opencv-python tqdm
```

## Usage

1. Prepare dataset in the following structure:
```
Expert_Train/
  Expert_TrainEval/
    Adult/
    Senior/ 
    Young/
PetFinder_All/
  Adult/
  Senior/
  Young/
```

2. Run training:
```bash
python Dogage/code/main.py
```

3. Visualize results:
```bash
python Dogage/code/visualize_results.py
```

## File Structure

```
Dogage/
  code/
    main.py - Main training script
    visualize_results.py - Visualization script
  results/
    mobilenet_v2/ - MobileNetV2 results
    efficientnet_b0/ - EfficientNetB0 results
    resnet18/ - ResNet18 results
```

## Requirements

- Python 3.7+
- PyTorch with GPU support
- Albumentations
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- tqdm
