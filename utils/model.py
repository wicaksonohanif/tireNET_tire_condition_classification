import torch
import json
from pathlib import Path
import torchvision.models as models
from torch import nn
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class MobileNetV2_CBAM(nn.Module):
    """MobileNetV2 with CBAM attention mechanism"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetV2_CBAM, self).__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            mobilenet = models.mobilenet_v2(weights=None)
        
        self.features = mobilenet.features
        
        # Add CBAM after last convolutional layer
        # MobileNetV2 last conv layer has 1280 channels
        self.cbam = CBAM(in_channels=1280, reduction=16)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(model_path, device='cpu'):
    """Load model dari checkpoint"""
    try:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model = MobileNetV2_CBAM(num_classes=2)
        
        model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {str(e)}")


def load_config(config_path):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_test_results(results_path):
    """Load test results"""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def get_model_info(config, test_results):
    """Prepare model information for display"""
    info = {
        'model_name': config.get('model_name', 'MobileNetV2_CBAM'),
        'image_size': config.get('image_size', 224),
        'num_classes': config.get('num_classes', 2),
        'batch_size': config.get('batch_size', 32),
        'learning_rate': config.get('learning_rate', 0.001),
        'optimizer': config.get('optimizer', 'Adam'),
        'scheduler': config.get('scheduler', 'ReduceLROnPlateau'),
        'training_epochs': test_results.get('training_epochs', 0),
        'test_accuracy': test_results.get('test_accuracy', 0),
        'best_val_accuracy': test_results.get('best_val_accuracy', 0),
    }
    return info
