import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transform(configs):
    """
    Create albumentations transform pipeline from config.
    
    Args:
        configs: List of transform configurations
        
    Returns:
        albumentations.Compose: Composed transform pipeline
    """
    transforms = []
    
    for transform in configs:
        transform_name = transform['name']
        transform_params = {k: v for k, v in transform.items() if k != 'name'}
        
        # Basic transforms
        if transform_name == 'Resize':
            transforms.append(A.Resize(
                height=transform_params.get('height', 224),
                width=transform_params.get('width', 224),
                interpolation=transform_params.get('interpolation', 1)
            ))
            
        elif transform_name == 'Normalize':
            transforms.append(A.Normalize(
                mean=transform_params.get('mean', [0.485, 0.456, 0.406]),
                std=transform_params.get('std', [0.229, 0.224, 0.225])
            ))
            
        # Geometric transforms
        elif transform_name == 'RandomHorizontalFlip':
            transforms.append(A.HorizontalFlip(p=transform_params.get('p', 0.5)))
            
        elif transform_name == 'RandomVerticalFlip':
            transforms.append(A.VerticalFlip(p=transform_params.get('p', 0.5)))
            
        elif transform_name == 'RandomRotation':
            transforms.append(A.Rotate(
                limit=transform_params.get('limit', 10),
                p=transform_params.get('p', 0.5),
                border_mode=transform_params.get('border_mode', 0)
            ))
            
        elif transform_name == 'ShiftScaleRotate':
            transforms.append(A.ShiftScaleRotate(
                shift_limit=transform_params.get('shift_limit', 0.1),
                scale_limit=transform_params.get('scale_limit', 0.1),
                rotate_limit=transform_params.get('rotate_limit', 15),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'RandomResizedCrop':
            transforms.append(A.RandomResizedCrop(
                size=(transform_params.get('height', 224), transform_params.get('width', 224)),
                scale=transform_params.get('scale', (0.8, 1.0)),
                ratio=transform_params.get('ratio', (0.75, 1.33)),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'CenterCrop':
            transforms.append(A.CenterCrop(
                height=transform_params.get('height', 224),
                width=transform_params.get('width', 224)
            ))
            
        # Color transforms
        elif transform_name == 'RandomGrayscale':
            transforms.append(A.ToGray(p=transform_params.get('p', 0.5)))
            
        elif transform_name == 'ColorJitter':
            transforms.append(A.ColorJitter(
                brightness=transform_params.get('brightness', 0.2),
                contrast=transform_params.get('contrast', 0.2),
                saturation=transform_params.get('saturation', 0.2),
                hue=transform_params.get('hue', 0.1),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'RandomBrightnessContrast':
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=transform_params.get('brightness_limit', 0.2),
                contrast_limit=transform_params.get('contrast_limit', 0.2),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'HueSaturationValue':
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=transform_params.get('hue_shift_limit', 10),
                sat_shift_limit=transform_params.get('sat_shift_limit', 20),
                val_shift_limit=transform_params.get('val_shift_limit', 10),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'CLAHE':
            transforms.append(A.CLAHE(
                clip_limit=transform_params.get('clip_limit', 2.0),
                tile_grid_size=transform_params.get('tile_grid_size', (8, 8)),
                p=transform_params.get('p', 0.5)
            ))
            
        # Noise and blur transforms
        elif transform_name == 'GaussianBlur':
            transforms.append(A.GaussianBlur(
                blur_limit=transform_params.get('blur_limit', (5, 9)),
                sigma_limit=transform_params.get('sigma_limit', (0.1, 2.0)),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'MotionBlur':
            transforms.append(A.MotionBlur(
                blur_limit=transform_params.get('blur_limit', 7),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'MedianBlur':
            transforms.append(A.MedianBlur(
                blur_limit=transform_params.get('blur_limit', 7),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'GaussNoise':
            transforms.append(A.GaussNoise(
                var_limit=transform_params.get('var_limit', (10.0, 50.0)),
                per_channel=transform_params.get('per_channel', True),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'ISONoise':
            transforms.append(A.ISONoise(
                color_shift=transform_params.get('color_shift', (0.01, 0.05)),
                intensity=transform_params.get('intensity', (0.1, 0.5)),
                p=transform_params.get('p', 0.5)
            ))
            
        # Dropout and cutout transforms
        elif transform_name == 'RandomSquareErasing':
            transforms.append(A.CoarseDropout(
                max_holes=transform_params.get('max_holes', 1),
                max_height=transform_params.get('max_height', 64),
                max_width=transform_params.get('max_width', 64),
                fill_value=transform_params.get('fill_value', 0),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'Cutout':
            transforms.append(A.CoarseDropout(
                max_holes=transform_params.get('max_holes', 8),
                max_height=transform_params.get('max_height', 16),
                max_width=transform_params.get('max_width', 16),
                fill_value=transform_params.get('fill_value', 0),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'GridDropout':
            transforms.append(A.GridDropout(
                ratio=transform_params.get('ratio', 0.5),
                unit_size_min=transform_params.get('unit_size_min', None),
                unit_size_max=transform_params.get('unit_size_max', None),
                holes_number_x=transform_params.get('holes_number_x', None),
                holes_number_y=transform_params.get('holes_number_y', None),
                shift_x=transform_params.get('shift_x', 0),
                shift_y=transform_params.get('shift_y', 0),
                random_offset=transform_params.get('random_offset', False),
                fill_value=transform_params.get('fill_value', 0),
                mask_fill_value=transform_params.get('mask_fill_value', None),
                p=transform_params.get('p', 0.5)
            ))
            
        # Weather and environment effects
        elif transform_name == 'RandomShadow':
            transforms.append(A.RandomShadow(
                shadow_roi=transform_params.get('shadow_roi', (0, 0.5, 1, 1)),
                num_shadows_lower=transform_params.get('num_shadows_lower', 1),
                num_shadows_upper=transform_params.get('num_shadows_upper', 2),
                shadow_dimension=transform_params.get('shadow_dimension', 5),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'RandomSunFlare':
            transforms.append(A.RandomSunFlare(
                flare_roi=transform_params.get('flare_roi', (0, 0, 1, 0.5)),
                angle_lower=transform_params.get('angle_lower', 0),
                angle_upper=transform_params.get('angle_upper', 1),
                num_flare_circles_lower=transform_params.get('num_flare_circles_lower', 6),
                num_flare_circles_upper=transform_params.get('num_flare_circles_upper', 10),
                src_radius=transform_params.get('src_radius', 400),
                src_color=transform_params.get('src_color', (255, 255, 255)),
                p=transform_params.get('p', 0.5)
            ))
            
        # Advanced transforms
        elif transform_name == 'ElasticTransform':
            transforms.append(A.ElasticTransform(
                alpha=transform_params.get('alpha', 1),
                sigma=transform_params.get('sigma', 50),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'GridDistortion':
            transforms.append(A.GridDistortion(
                num_steps=transform_params.get('num_steps', 5),
                distort_limit=transform_params.get('distort_limit', 0.3),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'OpticalDistortion':
            transforms.append(A.OpticalDistortion(
                distort_limit=transform_params.get('distort_limit', 0.05),
                shift_limit=transform_params.get('shift_limit', 0.05),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'Perspective':
            transforms.append(A.Perspective(
                scale=transform_params.get('scale', (0.05, 0.1)),
                keep_size=transform_params.get('keep_size', True),
                p=transform_params.get('p', 0.5)
            ))
            
        # Morphological operations
        elif transform_name == 'Downscale':
            transforms.append(A.Downscale(
                scale_min=transform_params.get('scale_min', 0.75),
                scale_max=transform_params.get('scale_max', 0.95),
                interpolation=transform_params.get('interpolation', 1),
                p=transform_params.get('p', 0.5)
            ))
            
        elif transform_name == 'ImageCompression':
            transforms.append(A.ImageCompression(
                quality_lower=transform_params.get('quality_lower', 60),
                quality_upper=transform_params.get('quality_upper', 100),
                compression_type=transform_params.get('compression_type', 1),
                p=transform_params.get('p', 0.5)
            ))
            
        else:
            print(f"Warning: Transform '{transform_name}' not implemented. Skipping...")
    
    # Always add ToTensor at the end if not already present
    if not any(isinstance(t, ToTensorV2) for t in transforms):
        transforms.append(ToTensorV2())
    
    return A.Compose(transforms)

def get_training_augmentations():
    """
    Get comprehensive training augmentations for facial age estimation.
    These are more aggressive augmentations suitable for training.
    """
    return [
        # Basic preprocessing
        dict(name='Resize', height=224, width=224),
        
        # Geometric augmentations (preserve facial structure)
        dict(name='RandomHorizontalFlip', p=0.5),
        dict(name='RandomRotation', limit=8, p=0.3),  # Small rotation to avoid distorting face
        dict(name='ShiftScaleRotate', shift_limit=0.1, scale_limit=0.1, rotate_limit=8, p=0.3),
        dict(name='RandomResizedCrop', height=224, width=224, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=0.3),
        dict(name='Perspective', scale=(0.02, 0.05), p=0.2),  # Subtle perspective changes
        
        # Color augmentations (lighting/contrast variations)
        dict(name='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        dict(name='ColorJitter', brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.4),
        dict(name='HueSaturationValue', hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.4),
        dict(name='CLAHE', clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        dict(name='RandomGrayscale', p=0.1),  # Lower probability to preserve color info
        
        # Noise and blur (simulate camera/environment conditions)
        dict(name='GaussianBlur', blur_limit=(3, 7), sigma_limit=(0.1, 1.5), p=0.2),
        dict(name='MotionBlur', blur_limit=5, p=0.1),
        dict(name='GaussNoise', var_limit=(5.0, 25.0), p=0.2),
        dict(name='ISONoise', color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=0.1),
        
        # Dropout/cutout (occlusion simulation)
        dict(name='Cutout', max_holes=3, max_height=16, max_width=16, p=0.2),
        dict(name='GridDropout', ratio=0.3, unit_size_min=8, unit_size_max=16, p=0.1),
        
        # Image quality degradation
        dict(name='ImageCompression', quality_lower=70, quality_upper=100, p=0.2),
        dict(name='Downscale', scale_min=0.8, scale_max=0.95, p=0.1),
        
        # Environmental effects (subtle)
        dict(name='RandomShadow', p=0.1),  # Very low probability for face images
        
        # Final normalization
        dict(name='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

def get_validation_augmentations():
    """
    Get minimal augmentations for validation/testing.
    Only basic preprocessing without random augmentations.
    """
    return [
        dict(name='Resize', height=224, width=224),
        dict(name='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

def get_light_augmentations():
    """
    Get light augmentations for facial age estimation.
    Conservative augmentations that preserve facial features.
    """
    return [
        dict(name='Resize', height=224, width=224),
        dict(name='RandomHorizontalFlip', p=0.5),
        dict(name='RandomBrightnessContrast', brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        dict(name='ColorJitter', brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
        dict(name='RandomRotation', limit=5, p=0.2),
        dict(name='GaussianBlur', blur_limit=(3, 5), sigma_limit=(0.1, 1.0), p=0.1),
        dict(name='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

def get_heavy_augmentations():
    """
    Get heavy augmentations for facial age estimation.
    More aggressive augmentations for robust training.
    """
    return [
        dict(name='Resize', height=224, width=224),
        dict(name='RandomHorizontalFlip', p=0.5),
        dict(name='RandomRotation', limit=12, p=0.4),
        dict(name='ShiftScaleRotate', shift_limit=0.15, scale_limit=0.15, rotate_limit=12, p=0.4),
        dict(name='RandomResizedCrop', height=224, width=224, scale=(0.8, 1.0), ratio=(0.85, 1.15), p=0.4),
        dict(name='ElasticTransform', alpha=1, sigma=30, p=0.2),
        dict(name='GridDistortion', num_steps=5, distort_limit=0.2, p=0.2),
        dict(name='OpticalDistortion', distort_limit=0.03, shift_limit=0.03, p=0.2),
        dict(name='RandomBrightnessContrast', brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        dict(name='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.08, p=0.5),
        dict(name='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),
        dict(name='CLAHE', clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
        dict(name='RandomGrayscale', p=0.2),
        dict(name='GaussianBlur', blur_limit=(3, 9), sigma_limit=(0.1, 2.0), p=0.3),
        dict(name='MotionBlur', blur_limit=7, p=0.2),
        dict(name='MedianBlur', blur_limit=5, p=0.1),
        dict(name='GaussNoise', var_limit=(10.0, 40.0), p=0.3),
        dict(name='ISONoise', color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.2),
        dict(name='RandomSquareErasing', max_holes=2, max_height=32, max_width=32, p=0.3),
        dict(name='Cutout', max_holes=5, max_height=20, max_width=20, p=0.3),
        dict(name='GridDropout', ratio=0.4, unit_size_min=6, unit_size_max=20, p=0.2),
        dict(name='ImageCompression', quality_lower=60, quality_upper=100, p=0.3),
        dict(name='Downscale', scale_min=0.7, scale_max=0.9, p=0.2),
        dict(name='RandomShadow', p=0.15),
        dict(name='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

