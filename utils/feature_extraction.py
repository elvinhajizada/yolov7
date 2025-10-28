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

def preprocess_image(image_path, image_size=640, device=None):
    """
    Preprocess the image for YOLOv7 by resizing and normalizing.
    Optionally move to specified device.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def detect_objects_yolov7(model, image_tensor, conf_threshold=0.1, iou_threshold=0.45):
    """
    Detect objects using YOLOv7 with given thresholds and extract objectness, class probabilities, and confidence scores.
    """
    # Move tensor to the same device as model
    image_tensor = image_tensor.to(next(model.parameters()).device)

    # Perform inference and get predictions
    with torch.no_grad():
        predictions = model(image_tensor)[0]  # Forward pass to get predictions

    # Apply Non-Maximum Suppression (NMS) with given thresholds
    detections = non_max_suppression(predictions, conf_threshold, iou_threshold)[0]
    
    if detections is not None:
        results = []
        for det in detections:  # Process each image's detections
            box = det[:4]  # Bounding box (xyxy)
            confidence = det[4].item()  # Confidence score
            objectness_score = det[5].item()
            class_id = int(det[6])  # Extract class ID
            
            
            # # Iterate over the raw predictions to find the best matching bounding box
            # for raw_pred in predictions[0]:
            #     raw_bbox = raw_pred[:4]  # Raw bounding box
            #     obj_score = raw_pred[4]  # Objectness score

            #     # if obj_score>0.1:
            #     #     print(obj_score, max(raw_pred[5:]))
            #     #     plt.figure()
            #     #     plt.bar(np.arange(len(raw_pred[5:])), raw_pred[5:].cpu())
            #     #     print(torch.sum(raw_pred[5:]))
            #     # Compute IoU between NMS box and raw predicted box
            #     # Ensure both boxes are 2D tensors with shape [1, 4]
            #     box_tensor = box.unsqueeze(0)  # Convert to [1, 4] tensor
            #     raw_bbox_tensor = raw_bbox.unsqueeze(0)  # Convert to [1, 4] tensor

            #     raw_bbox_tensor = xywh2xyxy(raw_bbox_tensor)
            #     iou = box_iou(box_tensor, raw_bbox_tensor)  # Convert to 1D tensor and compute IoU
                
            #     # If IoU is high, match the raw prediction with the NMS box
            #     if iou.item() > 0.9999:  # You can adjust the IoU threshold as needed
            #         print(f"Matched NMS Box: {box},")
            #         print(f"Raw Prediction Box: {raw_bbox},")
            #         print(f"IoU: {iou.item()}")
            #         objectness_score = obj_score.item()  # Assign objectness score from raw prediction
            #         print("Objectness score", objectness_score)
            #         # break  # Stop once the match is found

            # Append the detection result
            results.append([box, confidence, class_id, objectness_score])

        return results
    else:
        return []

def visualize_detections(image_path, detections, class_names, image_size=640):
    """
    Overlay detected bounding boxes on the input image, display objectness, class probabilities, 
    and final confidence in Jupyter.
    """
    # Read and resize the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))  # Resize for YOLOv7 input

    # Draw detections
    for detection in detections:
        box, confidence, class_id, obj_score= detection
        x1, y1, x2, y2 = box  # Bounding box coordinates
        class_conf = (confidence/obj_score)

        # Create label with class name, confidence, class probability, and objectness score
        label = (f"{class_names[class_id]}: {confidence:.2f}, Obj_conf: {obj_score:.2f}, Cls_conf: {class_conf:.2f} ")
        
        # Draw the bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Display the label with class info, confidence, and objectness above the box
        cv2.putText(image, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert BGR to RGB for matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Inspect YOLOv7 model architecture to identify good feature extraction layers
def inspect_model_layers(model, show_details=False):
    """
    Inspect the YOLOv7 model architecture to identify suitable layers for feature extraction.
    
    Args:
        model: YOLOv7 model
        show_details: Whether to show detailed layer information
    """
    print("YOLOv7 Model Architecture:")
    print("=" * 50)
    
    layer_count = 0
    conv_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.SiLU)):
            layer_count += 1
            
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
                
            if show_details:
                print(f"{layer_count:3d}. {name:40} -> {type(module).__name__}")
                if isinstance(module, nn.Conv2d):
                    print(f"     Input: {module.in_channels}, Output: {module.out_channels}, "
                          f"Kernel: {module.kernel_size}")
    
    print(f"\nTotal layers: {layer_count}")
    print(f"Convolutional layers: {len(conv_layers)}")
    
    # Suggest good layers for feature extraction
    print("\nSuggested layers for feature extraction:")
    if len(conv_layers) > 10:
        # Last few conv layers before detection heads
        for i, (name, module) in enumerate(conv_layers[-10:]):
            idx = len(conv_layers) - 10 + i
            print(f"  {idx+1:2d}. {name:40} - {module.out_channels} channels")
    
    return conv_layers

def extract_penultimate_features(model, image_tensor, target_layer_name='model.105'):
    """
    Extract penultimate layer features from YOLOv7 for feature representation.
    
    Args:
        model: YOLOv7 model
        image_tensor: Preprocessed image tensor
        target_layer_name: Name of the target layer (penultimate layer)
    
    Returns:
        features: Extracted feature vector from the penultimate layer
    """
    # Dictionary to store intermediate features
    features = {}
    
    def hook_fn(module, input, output):
        features['penultimate'] = output
    
    # Register hook on the target layer
    # For YOLOv7, the penultimate layer is typically before the final detection heads
    # Common layer names: 'model.105' (before final conv layers)
    target_layer = None
    for name, module in model.named_modules():
        if target_layer_name in name:
            target_layer = module
            break
    
    if target_layer is None:
        # Fallback: use the layer before the last few layers
        layers = list(model.named_modules())
        # Look for conv layers near the end but before detection heads
        for i, (name, module) in enumerate(reversed(layers)):
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)) and i > 5:
                target_layer = module
                print(f"Using fallback layer: {name}")
                break
    
    if target_layer is not None:
        handle = target_layer.register_forward_hook(hook_fn)
        
        # Move tensor to the same device as model
        image_tensor = image_tensor.to(next(model.parameters()).device)
        
        # Forward pass to extract features
        with torch.no_grad():
            _ = model(image_tensor)
        
        # Remove the hook
        handle.remove()
        
        # Extract and process features
        if 'penultimate' in features:
            feature_map = features['penultimate']
            
            # Handle case where feature_map might be a tuple (multiple outputs)
            if isinstance(feature_map, tuple):
                print(f"Feature map is tuple with {len(feature_map)} elements")
                # Find the best tensor in the tuple (largest spatial dimensions)
                best_tensor = None
                best_size = 0
                
                for i, elem in enumerate(feature_map):
                    if isinstance(elem, torch.Tensor):
                        if len(elem.shape) == 4:  # [batch, channels, height, width]
                            spatial_size = elem.shape[2] * elem.shape[3]
                            print(f"  Element {i}: shape {elem.shape}, spatial size: {spatial_size}")
                            if spatial_size > best_size:
                                best_size = spatial_size
                                best_tensor = elem
                        else:
                            print(f"  Element {i}: shape {elem.shape} (not 4D)")
                
                if best_tensor is not None:
                    feature_map = best_tensor
                    print(f"Selected tensor with shape {feature_map.shape}")
                else:
                    # Fallback: take the first tensor
                    feature_map = feature_map[0]
                    print(f"No good candidate found, using first element with shape {feature_map.shape}")
            
            # Ensure it's a tensor
            if not isinstance(feature_map, torch.Tensor):
                print(f"Feature map is not a tensor, type: {type(feature_map)}")
                return None
            
            print(f"Processing feature map with shape: {feature_map.shape}")
            
            # Global Average Pooling to get a feature vector
            if len(feature_map.shape) == 4:  # [batch, channels, height, width]
                feature_vector = torch.mean(feature_map, dim=[2, 3])  # Global average pooling
                print(f"Applied GAP, new shape: {feature_vector.shape}")
            elif len(feature_map.shape) == 3:  # [channels, height, width] - missing batch dim
                feature_vector = torch.mean(feature_map, dim=[1, 2])  # Global average pooling
                print(f"Applied GAP (3D), new shape: {feature_vector.shape}")
            elif len(feature_map.shape) == 2:  # [batch, features] - already flattened
                feature_vector = feature_map
                print(f"Already 2D, shape: {feature_vector.shape}")
            else:
                # For other shapes, try to flatten appropriately
                feature_vector = feature_map.view(feature_map.size(0), -1) if feature_map.size(0) == 1 else feature_map.flatten()
                print(f"Flattened to shape: {feature_vector.shape}")
            
            # Flatten if necessary
            if len(feature_vector.shape) > 1:
                feature_vector = feature_vector.flatten()
                print(f"Final flattened shape: {feature_vector.shape}")
            
            return feature_vector.cpu().numpy()
        else:
            print("Could not extract features from the specified layer")
            return None
    else:
        print("Target layer not found in the model")
        return None


def extract_features_with_detection(model, image_tensor, conf_threshold=0.1, iou_threshold=0.45):
    """
    Combined function to extract both detection results and penultimate features.
    Focuses on the main central bounding box for feature extraction.
    
    Returns:
        detections: List of detected objects
        features: Feature vector from penultimate layer
    """
    # Get detections first
    detections = detect_objects_yolov7(model, image_tensor, conf_threshold, iou_threshold)
    
    # Extract features
    features = extract_penultimate_features(model, image_tensor)
    
    return detections, features


def get_central_bbox_features(model, image_tensor, conf_threshold=0.1, iou_threshold=0.45):
    """
    Extract features specifically for the most central bounding box detection.
    
    Returns:
        central_detection: The most central detection
        features: Feature vector from penultimate layer
    """
    detections, features = extract_features_with_detection(model, image_tensor, conf_threshold, iou_threshold)
    
    if not detections:
        return None, features
    
    # Find the most central bounding box
    image_center = 320  # Assuming 640x640 input, center is at 320
    
    central_detection = None
    min_distance = float('inf')
    
    for detection in detections:
        box = detection[0]
        x1, y1, x2, y2 = box
        
        # Calculate center of bounding box
        bbox_center_x = (x1 + x2) / 2
        bbox_center_y = (y1 + y2) / 2
        
        # Calculate distance from image center
        distance = ((bbox_center_x - image_center) ** 2 + (bbox_center_y - image_center) ** 2) ** 0.5
        
        if distance < min_distance:
            min_distance = distance
            central_detection = detection
    
    return central_detection, features


def debug_feature_extraction(model, image_tensor, target_layer_name='model.105'):
    """
    Debug version to inspect what type of data we get from different layers.
    """
    features = {}
    
    def hook_fn(module, input, output):
        features['debug'] = output
        print(f"Hook triggered on layer: {target_layer_name}")
        print(f"Output type: {type(output)}")
        
        if isinstance(output, tuple):
            print(f"Output is tuple with {len(output)} elements:")
            for i, elem in enumerate(output):
                if isinstance(elem, torch.Tensor):
                    print(f"  Element {i}: Tensor shape {elem.shape}, dtype {elem.dtype}")
                    if len(elem.shape) == 4 and elem.shape[2] > 1 and elem.shape[3] > 1:
                        print(f"    -> Good candidate for feature extraction (spatial dimensions)")
                else:
                    print(f"  Element {i}: {type(elem)}")
        elif isinstance(output, torch.Tensor):
            print(f"Output is tensor with shape: {output.shape}, dtype {output.dtype}")
            if len(output.shape) == 4 and output.shape[2] > 1 and output.shape[3] > 1:
                print(f"  -> Good candidate for feature extraction (spatial dimensions)")
        else:
            print(f"Output is {type(output)}")
    
    # Find and hook the target layer
    target_layer = None
    for name, module in model.named_modules():
        if target_layer_name in name:
            target_layer = module
            print(f"Found target layer: {name} ({type(module).__name__})")
            break
    
    if target_layer is not None:
        handle = target_layer.register_forward_hook(hook_fn)
        
        # Move tensor to the same device as model
        image_tensor = image_tensor.to(next(model.parameters()).device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(image_tensor)
        
        # Remove the hook
        handle.remove()
        
        return features.get('debug', None)
    else:
        print(f"Target layer '{target_layer_name}' not found")
        return None

# Batch feature extraction for classifier training
def extract_features_batch(image_paths, model, conf_threshold=0.1, max_images=None):
    """
    Extract penultimate layer features from multiple images for classifier training.
    
    Args:
        image_paths: List of image file paths
        model: YOLOv7 model
        conf_threshold: Confidence threshold for detection
        max_images: Maximum number of images to process (None for all)
    
    Returns:
        features_list: List of feature vectors
        detections_list: List of detection results
        valid_paths: List of valid image paths (with detections)
    """
    features_list = []
    detections_list = []
    valid_paths = []
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    for i, image_path in enumerate(image_paths):
        try:
            # Preprocess the image
            image_tensor = preprocess_image(image_path, image_size=640)
            
            # Extract features for central bounding box
            central_detection, features = get_central_bbox_features(model, image_tensor, conf_threshold)
            
            if features is not None and central_detection is not None:
                features_list.append(features)
                detections_list.append(central_detection)
                valid_paths.append(image_path)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue
    
    return np.array(features_list), detections_list, valid_paths