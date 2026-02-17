import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def preprocess_image(image_path_or_array, image_size=224, device='cpu'):
    """
    Preprocess image untuk model inference
    
    Args:
        image_path_or_array: Path ke image atau numpy array
        image_size: Target image size
        device: Device untuk tensor (cpu atau cuda)
    
    Returns:
        Preprocessed tensor
    """
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load image
    if isinstance(image_path_or_array, str):
        image = Image.open(image_path_or_array).convert('RGB')
    elif isinstance(image_path_or_array, np.ndarray):
        image = Image.fromarray(image_path_or_array).convert('RGB')
    else:
        image = image_path_or_array.convert('RGB')
    
    # Apply transform
    tensor = transform(image).unsqueeze(0).to(device)
    
    return tensor, image


def predict(model, image_tensor, device='cpu'):
    """
    Predict class dan confidence
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        device: Device untuk inference
    
    Returns:
        predictions: softmax output untuk semua classes
        predicted_class: predicted class index
        confidence: confidence score untuk predicted class
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predictions = probabilities[0].cpu().numpy()
        predicted_class = predicted.item()
        confidence_score = confidence.item()
    
    return predictions, predicted_class, confidence_score


def get_class_name(class_idx, class_names=None):
    """Get class name dari index"""
    if class_names is None:
        class_names = {0: "Good", 1: "Defective"}
    return class_names.get(class_idx, f"Class {class_idx}")
