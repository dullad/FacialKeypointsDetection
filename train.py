import os
import yaml
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models as models

# 1. 模型定义
class KeypointModel(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True, dropout=0.1):
        super(KeypointModel, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.dropout = dropout
        if self.backbone_name.startswith('resnet'):
            # --- ResNet 系列骨干网络 ---
            if self.backbone_name == 'resnet18':
                self.backbone = models.resnet18(pretrained=pretrained)
                num_features = self.backbone.fc.in_features
            elif self.backbone_name == 'resnet34':
                self.backbone = models.resnet34(pretrained=pretrained)
                num_features = self.backbone.fc.in_features
            else:
                raise ValueError(f"不支持的ResNet模型: {backbone_name}")

            # 修改输入层以适应单通道（灰度）图像
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1, original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=False
            )
            # 如果使用预训练权重，则复制并平均权重以适应单通道
            if pretrained:
                self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

            # 替换输出层以适应30个关键点
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(num_features, 30)
            )

        elif self.backbone_name.startswith('efficientnet'):
            # --- EfficientNet 系列骨干网络 ---
            if self.backbone_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                num_features = self.backbone.classifier[1].in_features
            elif self.backbone_name == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(pretrained=pretrained)
                num_features = self.backbone.classifier[1].in_features
            else:
                raise ValueError(f"不支持的EfficientNet模型: {backbone_name}")

            # 修改输入层
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                1, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            if pretrained:
                self.backbone.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
            
            # 替换输出层
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout, inplace=True),
                nn.Linear(num_features, 30),
            )
        else:
            raise ValueError(f"未知的backbone_name: {backbone_name}")

    def forward(self, x):
        x = self.backbone(x)
        return x

class FaceKeypointsDataset(Dataset):
    def __init__(self, images, keypoints=None, transform=None):
        self.images = images
        self.keypoints = keypoints
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape(96, 96)

        if self.keypoints is not None:
            kps_px = self.keypoints[idx].reshape(-1, 2).astype(np.float32)
            if self.transform:
                transformed = self.transform(image=image, keypoints=kps_px.tolist())
                image = transformed['image']
                kps_px = np.array(transformed['keypoints'], dtype=np.float32)

            kps_px = np.clip(kps_px, 0.0, 96.0)

            kps_norm = (kps_px - 48.0) / 48.0
            kps_tensor = torch.from_numpy(kps_norm).float().flatten()
            return image, kps_tensor
        else:
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            return image

# 缺失值智能填充
def smart_fill_na(df):
    # 定义对称特征对
    keypoint_pairs = [
        ('left_eye_center_x', 'right_eye_center_x'),
        ('left_eye_inner_corner_x', 'right_eye_inner_corner_x'),
        ('left_eye_outer_corner_x', 'right_eye_outer_corner_x'),
        ('left_eyebrow_inner_end_x', 'right_eyebrow_inner_end_x'),
        ('left_eyebrow_outer_end_x', 'right_eyebrow_outer_end_x'),
        ('mouth_left_corner_x', 'mouth_right_corner_x'),
    ]
    
    # 对称填充
    for left_x, right_x in keypoint_pairs:
        left_y = left_x.replace('_x', '_y')
        right_y = right_x.replace('_x', '_y')
        
        # 左边缺失，右边存在
        left_nan = df[left_x].isnull() & df[right_x].notnull()
        df.loc[left_nan, left_x] = 96 - df.loc[left_nan, right_x]
        df.loc[left_nan, left_y] = df.loc[left_nan, right_y]
        
        # 右边缺失，左边存在
        right_nan = df[right_x].isnull() & df[left_x].notnull()
        df.loc[right_nan, right_x] = 96 - df.loc[right_nan, left_x]
        df.loc[right_nan, right_y] = df.loc[right_nan, left_y]
        
    # 均值填充剩余的缺失值
    df.fillna(df.mean(), inplace=True)
    return df

# 数据加载和预处理函数
def load_data(data_path, is_train=True):
    df = pd.read_csv(data_path)
    df['Image'] = df['Image'].apply(lambda img: np.fromstring(img, sep=' '))
    X = np.vstack(df['Image'].values)
    X = X.astype(np.float32).reshape(-1, 96, 96, 1)
    
    if is_train:
        # 智能填充缺失值
        keypoints_df = smart_fill_na(df.drop('Image', axis=1))
        y = keypoints_df.values
        return X, y, keypoints_df
    else:
        return X

# 主训练函数
def main():
    # --- 设置和初始化 ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志记录
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(log_dir, 'training.log')),
                            logging.StreamHandler()
                        ])
    
    logging.info("--- 训练开始 ---")
    logging.info(f"配置加载于: {log_dir}")
    
    # 保存配置文件和模型配置
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # --- 设备选择 ---
    gpu_id = config['run_params']['gpu_id']
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id != -1 else "cpu")
    logging.info(f"使用设备: {device}")

    # --- 数据准备 ---
    logging.info("加载和预处理数据...")
    X_train_orig, y_train_px, train_cols_df = load_data(config['run_params']['train_csv'], is_train=True)
    X_test = load_data(config['run_params']['test_csv'], is_train=False)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_orig, y_train_px, test_size=0.2, random_state=42
    )

    # 数据增强
    train_transform = A.Compose([
        # A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=10, p=0.3),
        A.Normalize(mean=[0.5], std=[0.5]), # 像素归一到[-1, 1]，是否有用？
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    val_transform = A.Compose([
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    train_dataset = FaceKeypointsDataset(X_train, y_train, transform=train_transform)
    val_dataset = FaceKeypointsDataset(X_val, y_val, transform=val_transform)
    
    batch_size = config['model_params']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    logging.info("数据准备完成。")

    # --- 模型、损失函数、优化器 ---
    backbone_name = config['model_params'].get('backbone_name', 'resnet18')
    dropout = config['model_params'].get('dropout', 0.2)
    logging.info(f"使用模型骨干: {backbone_name}")
    model = KeypointModel(backbone_name=backbone_name, dropout=dropout).to(device)
    with open(os.path.join(log_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model))

    criterion = nn.SmoothL1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=config['model_params']['learning_rate'], weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=config['model_params']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 训练循环与早停 ---
    epochs = config['model_params']['epochs']
    patience = config['early_stopping']['patience']
    min_delta = config['early_stopping']['min_delta']
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

    logging.info("--- 开始训练循环 ---")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        scheduler.step(avg_val_loss)

        # 早停和模型保存逻辑
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
            logging.info(f"验证损失改善，保存模型。最佳Val Loss: {best_val_loss:.6f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            logging.info(f"连续 {patience} 个 epochs 验证损失没有改善，触发早停。")
            break
    
    logging.info("--- 训练结束 ---")

    # --- 预测与生成提交文件 ---
    logging.info("加载最佳模型并进行预测...")
    model_best = KeypointModel(backbone_name=backbone_name).to(device)
    model_best.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')))
    model_best.eval()

    test_dataset = FaceKeypointsDataset(X_test, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model_best(inputs)
            all_preds.append(outputs.cpu().numpy())
    
    pred_normalized = np.concatenate(all_preds, axis=0)
    
    # 反归一化预测结果
    pred = pred_normalized * 48.0 + 48.0
    pred = np.clip(pred, 0.0, 96.0) 

    logging.info("根据 IdLookupTable 生成提交文件...")
    lookid_data = pd.read_csv(config['run_params']['lookup_table_csv'])
    feature_names = list(train_cols_df.columns)
    feature_to_idx = {name: i for i, name in enumerate(feature_names)}

    image_ids_to_lookup = lookid_data['ImageId']
    features_to_lookup = lookid_data['FeatureName']

    locations = [pred[image_id - 1][feature_to_idx[feature_name]] 
                 for image_id, feature_name in zip(image_ids_to_lookup, features_to_lookup)]

    submission = pd.DataFrame({
        "RowId": lookid_data['RowId'],
        "Location": locations
    })
    
    submission_filename = f"{best_val_loss:.6f}_submission.csv"
    submission_path = os.path.join(log_dir, submission_filename)
    submission.to_csv(submission_path, index=False)
    
    logging.info(f"提交文件已保存至: {submission_path}")
    logging.info("--- 全部完成 ---")

if __name__ == '__main__':
    main()