# 🎯 ROI-BASED FEATURE EXTRACTION FOR SPECIFIC BOUNDING BOXES
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import os
import shutil

import torchvision.transforms as transforms
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
import torch.nn as nn

from utils.general import non_max_suppression, xywh2xyxy, box_iou  # Import the IoU function

# Import from our feature_extraction module - avoid circular imports
try:
    from .feature_extraction import preprocess_image, detect_objects_yolov7
except ImportError:
    # Fallback for relative imports
    from feature_extraction import preprocess_image, detect_objects_yolov7

def extract_bbox_features(model, image_tensor, bbox, target_layer_name='model.99'):
    """
    Extract features specifically from a bounding box region in the feature map.
    
    Args:
        model: YOLOv7 model
        image_tensor: Preprocessed image tensor [1, 3, 640, 640]
        bbox: Bounding box [x1, y1, x2, y2] in original image coordinates
        target_layer_name: Name of the target layer
    
    Returns:
        features: Feature vector extracted from the bounding box region only
    """
    features = {}
    
    def hook_fn(module, input, output):
        features['roi'] = output
    
    # Find target layer
    target_layer = None
    for name, module in model.named_modules():
        if target_layer_name in name:
            target_layer = module
            break
    
    if target_layer is None:
        print(f"Target layer '{target_layer_name}' not found")
        return None
    
    # Register hook
    handle = target_layer.register_forward_hook(hook_fn)
    
    try:
        # Move tensor to device
        image_tensor = image_tensor.to(next(model.parameters()).device)
        
        # Forward pass to get feature maps
        with torch.no_grad():
            _ = model(image_tensor)
        
        # Remove hook
        handle.remove()
        
        if 'roi' not in features:
            print("Failed to capture features")
            return None
        
        feature_map = features['roi']
        
        # Handle tuple outputs
        if isinstance(feature_map, tuple):
            # Select best tensor from tuple
            best_tensor = None
            best_size = 0
            for elem in feature_map:
                if isinstance(elem, torch.Tensor) and len(elem.shape) == 4:
                    spatial_size = elem.shape[2] * elem.shape[3]
                    if spatial_size > best_size:
                        best_size = spatial_size
                        best_tensor = elem
            feature_map = best_tensor if best_tensor is not None else feature_map[0]
        
        if not isinstance(feature_map, torch.Tensor):
            print(f"Feature map is not a tensor: {type(feature_map)}")
            return None
        
        # Get feature map dimensions
        batch_size, channels, feat_h, feat_w = feature_map.shape
        
        # Convert bbox coordinates from image space (640x640) to feature map space
        x1, y1, x2, y2 = bbox
        
        # Calculate scaling factors
        scale_x = feat_w / 640.0  # feature_width / image_width
        scale_y = feat_h / 640.0  # feature_height / image_height
        
        # Scale bbox to feature map coordinates
        feat_x1 = int(x1 * scale_x)
        feat_y1 = int(y1 * scale_y)
        feat_x2 = int(x2 * scale_x)
        feat_y2 = int(y2 * scale_y)
        
        # Ensure coordinates are within bounds
        feat_x1 = max(0, min(feat_x1, feat_w - 1))
        feat_y1 = max(0, min(feat_y1, feat_h - 1))
        feat_x2 = max(feat_x1 + 1, min(feat_x2, feat_w))
        feat_y2 = max(feat_y1 + 1, min(feat_y2, feat_h))
        
        print(f"Original bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"Feature map shape: {feature_map.shape}")
        print(f"Feature bbox: [{feat_x1}, {feat_y1}, {feat_x2}, {feat_y2}]")
        
        # Extract ROI from feature map
        roi_features = feature_map[:, :, feat_y1:feat_y2, feat_x1:feat_x2]
        print(f"ROI features shape: {roi_features.shape}")
        
        # Apply Global Average Pooling to ROI
        if roi_features.shape[2] > 0 and roi_features.shape[3] > 0:
            # Average pool over spatial dimensions
            pooled_features = torch.mean(roi_features, dim=[2, 3])  # [1, channels]
            feature_vector = pooled_features.flatten()  # [channels]
            
            print(f"Final feature vector shape: {feature_vector.shape}")
            return feature_vector.cpu().numpy()
        else:
            print("ROI has zero spatial dimensions")
            return None
            
    except Exception as e:
        print(f"Error in ROI feature extraction: {str(e)}")
        if 'handle' in locals():
            handle.remove()
        return None


def extract_all_bbox_features(model, image_tensor, detections, target_layer_name='model.99'):
    """
    Extract features for all detected bounding boxes.
    
    Args:
        model: YOLOv7 model
        image_tensor: Preprocessed image tensor
        detections: List of detections from detect_objects_yolov7()
        target_layer_name: Target layer name
    
    Returns:
        bbox_features: List of feature vectors, one per bounding box
        bbox_info: List of bounding box information
    """
    bbox_features = []
    bbox_info = []
    
    for i, detection in enumerate(detections):
        bbox = detection[0]  # [x1, y1, x2, y2]
        confidence = detection[1]
        class_id = detection[2]
        
        # Convert tensor bbox to list if needed
        if hasattr(bbox, 'cpu'):
            bbox = bbox.cpu().numpy().tolist()
        
        # Extract features for this specific bounding box
        features = extract_bbox_features(model, image_tensor, bbox, target_layer_name)
        
        if features is not None:
            bbox_features.append(features)
            bbox_info.append({
                'bbox': bbox,
                'confidence': confidence,
                'class_id': class_id,
                'detection_idx': i
            })
            print(f"✓ Extracted features for bbox {i}: {features.shape}")
        else:
            print(f"✗ Failed to extract features for bbox {i}")
    
    return bbox_features, bbox_info


def get_central_bbox_roi_features(model, image_tensor, conf_threshold=0.1, target_layer_name='model.99'):
    """
    Get ROI features for the most central bounding box.
    
    Returns:
        central_detection: Info about central detection
        roi_features: Feature vector from central bounding box region only
    """
    # Get detections
    detections = detect_objects_yolov7(model, image_tensor, conf_threshold)
    
    if not detections:
        print("No detections found")
        return None, None
    
    # Find most central detection
    image_center = 320  # Center of 640x640 image
    central_detection = None
    min_distance = float('inf')
    
    for detection in detections:
        bbox = detection[0]
        if hasattr(bbox, 'cpu'):
            bbox = bbox.cpu().numpy()
        
        x1, y1, x2, y2 = bbox
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        distance = ((bbox_center_x - image_center) ** 2 + (bbox_center_y - image_center) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            central_detection = detection
    
    if central_detection is None:
        return None, None
    
    # Extract ROI features for central detection
    central_bbox = central_detection[0]
    if hasattr(central_bbox, 'cpu'):
        central_bbox = central_bbox.cpu().numpy().tolist()
    
    roi_features = extract_bbox_features(model, image_tensor, central_bbox, target_layer_name)
    
    return central_detection, roi_features


def extract_roi_features_batch(image_paths, model, conf_threshold=0.1, target_layer='model.99', 
                               max_images=None, central_only=True):
    """
    Extract ROI-based features from multiple images for classifier training.
    
    Args:
        image_paths: List of image file paths
        model: YOLOv7 model
        conf_threshold: Confidence threshold for detection
        target_layer: Layer to extract features from
        max_images: Maximum number of images to process
        central_only: If True, extract only central bbox features; if False, extract all bbox features
    
    Returns:
        features_list: List of feature vectors (one per bounding box)
        bbox_info_list: List of bounding box information
        valid_paths: List of successfully processed image paths
    """
    features_list = []
    bbox_info_list = []
    valid_paths = []
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"🎯 Processing {len(image_paths)} images with ROI-based extraction...")
    print(f"Mode: {'Central bbox only' if central_only else 'All bboxes'}")
    
    for i, image_path in enumerate(image_paths):
        try:
            # Preprocess image
            image_tensor = preprocess_image(image_path, image_size=640)
            
            if central_only:
                # Extract features for central bounding box only
                central_detection, roi_features = get_central_bbox_roi_features(
                    model, image_tensor, conf_threshold, target_layer
                )
                
                if roi_features is not None and central_detection is not None:
                    features_list.append(roi_features)
                    
                    # Convert bbox to list format
                    bbox = central_detection[0]
                    if hasattr(bbox, 'cpu'):
                        bbox = bbox.cpu().numpy().tolist()
                    
                    bbox_info_list.append({
                        'image_path': image_path,
                        'bbox': bbox,
                        'confidence': central_detection[1],
                        'class_id': central_detection[2],
                        'is_central': True
                    })
                    valid_paths.append(image_path)
                
            else:
                # Extract features for all bounding boxes
                detections = detect_objects_yolov7(model, image_tensor, conf_threshold)
                bbox_features, bbox_info = extract_all_bbox_features(
                    model, image_tensor, detections, target_layer
                )
                
                for features, info in zip(bbox_features, bbox_info):
                    features_list.append(features)
                    info['image_path'] = image_path
                    info['is_central'] = False
                    bbox_info_list.append(info)
                    
                if bbox_features:  # Only add path if we got some features
                    valid_paths.append(image_path)
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images, got {len(features_list)} feature vectors")
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    print(f"\n✅ Completed! Extracted {len(features_list)} ROI feature vectors from {len(valid_paths)} images")
    
    return np.array(features_list), bbox_info_list, valid_paths

