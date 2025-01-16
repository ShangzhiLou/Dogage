import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import copy

# Set path, assume data is already extracted
dataset_path = os.path.join(".")

print("Path to dataset files:", dataset_path)

def load_data(root_dir):
    """加载数据集，返回 DataFrame"""
    data = []
    for dataset_type in ['Expert_Train', 'PetFinder_All']:
        if dataset_type == 'Expert_Train':
            dataset_path = os.path.join(root_dir, dataset_type, 'Expert_TrainEval')
        else:
            dataset_path = os.path.join(root_dir, dataset_type)
        for age_group in ['Adult', 'Senior', 'Young']:
             age_path = os.path.join(dataset_path, age_group)
             if os.path.exists(age_path):  # 检查路径是否存在
                 for filename in tqdm(os.listdir(age_path), desc=f"Loading {dataset_type} - {age_group}"):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 检查文件类型
                        filepath = os.path.join(age_path, filename)
                        data.append({'filepath': filepath, 'label': age_group})
    return pd.DataFrame(data)

def split_data(df):
    """划分数据集"""
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    # 从测试集中划分验证集
    val_df, test_df = train_test_split(test_df, test_size=0.5, stratify=test_df['label'], random_state=42)
    return train_df, val_df, test_df

def get_transforms(train=True, image_size=224):
    """定义数据增强"""
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# 人脸检测函数
def detect_face(image):
     """检测图像中的人脸，并返回裁剪后的图像"""
     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
     if len(faces) > 0:
         x, y, w, h = faces[0]
         x = max(0,x-20) # 适当扩大检测框，防止头部截断
         y = max(0,y-20)
         w = min(image.shape[1]-x,w+40)
         h = min(image.shape[0]-y,h+40)
         cropped_face = image[y:y+h, x:x+w]
         return cropped_face
     else:
         return image

class DogAgeDataset(torch.utils.data.Dataset):
    """自定义数据集"""
    def __init__(self, df, transform=None, use_face_detection=True):
        self.df = df
        self.transform = transform
        self.labels = {'Young': 0, 'Adult': 1, 'Senior': 2}
        self.use_face_detection = use_face_detection
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        filepath = self.df.iloc[idx]['filepath']
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.use_face_detection:
            image = detect_face(image)
        label = self.labels[self.df.iloc[idx]['label']]
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return image, torch.tensor(label, dtype=torch.long) #确保label是long tensor

def create_model(model_name='mobilenet_v2', num_classes=3, pretrained=True):
    """创建模型"""
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # 修改最后一层
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model

def train_model(train_df, val_df, model_name='mobilenet_v2', image_size=224, batch_size=32, num_epochs=20, model_path='best_dog_model.pth', use_gpu=True, use_face_detection=True):
    """训练模型"""
    # 模型
    model = create_model(model_name=model_name, num_classes=3)
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # 数据加载器
    train_transforms = get_transforms(train=True, image_size=image_size)
    val_transforms = get_transforms(train=False, image_size=image_size)
    train_dataset = DogAgeDataset(train_df, transform=train_transforms, use_face_detection = use_face_detection)
    val_dataset = DogAgeDataset(val_df, transform=val_transforms, use_face_detection = use_face_detection)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 训练循环
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            if use_gpu and torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # 验证
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                if use_gpu and torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step(epoch_loss)
    print(f"Best val Acc: {best_acc:4f}")
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_path)
    return model

def eval_model(model, test_df, image_size=224, batch_size=32, use_gpu=True, use_face_detection=True):
    """测试模型"""
    test_transforms = get_transforms(train=False, image_size=image_size)
    test_dataset = DogAgeDataset(test_df, transform=test_transforms, use_face_detection=use_face_detection)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            if use_gpu and torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(all_labels, all_preds))
    print('\nConfusion Matrix:')
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':

    root_dir = os.path.join(dataset_path)  # Use the extracted dataset path

    df = load_data(root_dir)
    train_df, val_df, test_df = split_data(df)

    image_size = 224
    batch_size = 32
    num_epochs = 3

    # 模型训练和评估
    model_names = ['mobilenet_v2', 'efficientnet_b0', 'resnet18']
    for model_name in model_names:
        model_path = f'best_dog_model_{model_name}.pth'
        trained_model = train_model(train_df, val_df, model_name=model_name, image_size=image_size, batch_size=batch_size, num_epochs=num_epochs, model_path=model_path, use_face_detection=True)
        eval_model(trained_model, test_df, image_size=image_size, batch_size=batch_size, use_face_detection=True)