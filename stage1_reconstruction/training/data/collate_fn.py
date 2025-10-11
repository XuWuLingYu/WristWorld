"""
Custom collate function for handling variable-sized point clouds in VGGT DROID dataset.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional


def vggt_droid_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for VGGT DROID dataset that handles variable-sized point clouds.
    
    Args:
        batch: List of sample dictionaries from the dataset
        
    Returns:
        Collated batch dictionary with proper handling of variable-sized data
    """
    if not batch:
        return {}
    
    # Extract batch size
    batch_size = len(batch)
    
    # Initialize the collated batch
    collated = {}
    
    # Handle fixed-size data (can be safely stacked)
    # Note: cam_points and world_points are moved here because training code expects them as tensors
    fixed_size_keys = ["images", "depths", "depth_nan_masks", "extrinsics", "intrinsics", 
                       "cam_points", "world_points", "point_masks", "original_sizes"]
    if 'depths' not in batch[0].keys():
        fixed_size_keys.remove('depths')
        fixed_size_keys.remove('depth_nan_masks')
        fixed_size_keys.remove("world_points")
        fixed_size_keys.remove("point_masks")
        
    for key in fixed_size_keys:
        if key in batch[0]:
            # Stack the data from all samples
            stacked_data = []
            for sample in batch:
                if key == "images":
                    # Special handling for images: convert from list of (H,W,3) to (S,3,H,W)
                    if isinstance(sample[key], list):
                        # Stack images into (S, H, W, 3)
                        images_array = np.stack(sample[key]).astype(np.float32)
                        # Convert to tensor and permute to (S, 3, H, W)
                        images_tensor = torch.from_numpy(images_array).permute(0, 3, 1, 2)
                        # Normalize from [0, 255] to [0, 1]
                        images_tensor = images_tensor.to(torch.get_default_dtype()).div(255)
                        stacked_data.append(images_tensor)
                    else:
                        stacked_data.append(torch.as_tensor(sample[key]))
                elif isinstance(sample[key], list):
                    # For other list data (like depths), stack within each sample
                    stacked_data.append(torch.stack([torch.as_tensor(item) for item in sample[key]]))
                elif key in ["wrist_extrinsics", "wrist_intrinsics"]:
                    # Special handling for wrist parameters: add sequence dimension
                    stacked_data.append(torch.as_tensor(sample[key]).unsqueeze(0))  # Add S=1 dimension
                elif key == "depth_nan_masks":
                    # Handle depth NaN masks as boolean tensors
                    if isinstance(sample[key], list):
                        stacked_data.append(torch.stack([torch.as_tensor(item, dtype=torch.bool) for item in sample[key]]))
                    else:
                        stacked_data.append(torch.as_tensor(sample[key], dtype=torch.bool))
                else:
                    stacked_data.append(torch.as_tensor(sample[key]))
            collated[key] = torch.stack(stacked_data)
    
    # Handle variable-size point cloud data (cannot be stacked) - these remain as lists
    variable_size_keys = ["point_cloud", "point_colors"]
    
    for key in variable_size_keys:
        if key in batch[0]:
            # Keep as list since sizes can vary
            collated[key] = [sample[key] for sample in batch]
    
    # Handle scalar/string data
    scalar_keys = ["seq_name", "ids", "frame_num", "frame_id"]
    
    for key in scalar_keys:
        if key in batch[0]:
            if key in ["seq_name", "frame_id"]:
                # For string data, keep as list
                collated[key] = [sample[key] for sample in batch]
            elif key == "ids":
                # Handle ids specially - they can be numpy arrays or lists
                values = [sample[key] for sample in batch]
                # Convert all values to tensors, regardless of their original type
                tensor_values = []
                for val in values:
                    if isinstance(val, np.ndarray):
                        tensor_values.append(torch.from_numpy(val))
                else:
                        tensor_values.append(torch.as_tensor(val))
                collated[key] = torch.stack(tensor_values)
            else:
                # For other scalar data, try to stack
                values = [sample[key] for sample in batch]
                if isinstance(values[0], (int, float)):
                    collated[key] = torch.tensor(values)
                else:
                    collated[key] = torch.stack([torch.as_tensor(val) for val in values])
    
    # Handle wrist-related data if present
    wrist_keys = ["wrist_extrinsics", "wrist_intrinsics"]
    for key in wrist_keys:
        if key in batch[0]:
            stacked_data = []
            for sample in batch:
                # Add sequence dimension for wrist parameters
                stacked_data.append(torch.as_tensor(sample[key]).unsqueeze(0))  # Add S=1 dimension
            collated[key] = torch.stack(stacked_data)
    
    # Handle wrist image data if present
    wrist_image_keys = ["wrist_image", "wrist_image_path", "wrist_image_shape"]
    for key in wrist_image_keys:
        if key in batch[0]:
            if key == "wrist_image":
                # Handle wrist_image as tensor or list
                stacked_data = []
                for sample in batch:
                    if sample[key] is not None:
                        if isinstance(sample[key], np.ndarray):
                            stacked_data.append(torch.from_numpy(sample[key]))
                        else:
                            stacked_data.append(torch.as_tensor(sample[key]))
                collated[key] = torch.stack(stacked_data)
            elif key == "wrist_image_path":
                # Keep as list for string paths
                collated[key] = [sample[key] for sample in batch]
            elif key == "wrist_image_shape":
                # Keep as list for shape tuples
                collated[key] = [sample[key] for sample in batch]
    
    return collated


def vggt_droid_single_view_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Deprecated: Keep for backward compatibility but route to unified collate
    return vggt_droid_collate_fn(batch)


def pad_point_clouds(point_clouds: List[np.ndarray], max_points: Optional[int] = None) -> torch.Tensor:
    """
    Pad variable-size point clouds to create a tensor.
    
    Args:
        point_clouds: List of point clouds with shape (N_i, 3)
        max_points: Maximum number of points. If None, use the max from the batch
        
    Returns:
        Padded tensor of shape (batch_size, max_points, 3)
    """
    if max_points is None:
        max_points = max(pc.shape[0] for pc in point_clouds)
    
    assert max_points is not None
    batch_size = len(point_clouds)
    padded = torch.zeros(batch_size, max_points, 3)
    
    for i, pc in enumerate(point_clouds):
        n_points = min(pc.shape[0], max_points)
        padded[i, :n_points] = torch.from_numpy(pc[:n_points])
    
    return padded


def create_padding_mask(point_clouds: List[np.ndarray], max_points: Optional[int] = None) -> torch.Tensor:
    """
    Create padding mask for variable-size point clouds.
    
    Args:
        point_clouds: List of point clouds with shape (N_i, 3)
        max_points: Maximum number of points. If None, use the max from the batch
        
    Returns:
        Boolean mask tensor of shape (batch_size, max_points)
    """
    if max_points is None:
        max_points = max(pc.shape[0] for pc in point_clouds)
    
    assert max_points is not None
    batch_size = len(point_clouds)
    mask = torch.zeros(batch_size, max_points, dtype=torch.bool)
    
    for i, pc in enumerate(point_clouds):
        n_points = min(pc.shape[0], max_points)
        mask[i, :n_points] = True
    
    return mask 