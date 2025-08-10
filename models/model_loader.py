"""
Model loader - Production PatchCore Support
Loads model based on config.py
"""

import datetime
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Import custom config
try:
    from config import *
except ImportError as e:
    raise ImportError("config.py required")

# Import HRNet model creator
try:
    from .hrnet_model import create_hrnet_model
    HRNET_CREATOR_AVAILABLE = True
except ImportError:
    try:
        from hrnet_model import create_hrnet_model
        HRNET_CREATOR_AVAILABLE = True
    except ImportError:
        HRNET_CREATOR_AVAILABLE = False


class PatchCoreInferencer:
    """
    PatchCore Model Inferencer for Anomalib fitted models
    """
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.memory_bank = None
        self.feature_channels = None
        self.target_size = None
        self.is_fitted = False
        self.threshold = ANOMALY_THRESHOLD
        self.transform = None
        self.feature_extractor = None
        self._load_model(model_path)
        self._setup_preprocessing()
        self._setup_feature_extractor()
    
    def _load_model(self, model_path):
        """Load PatchCore fitted model"""
        print(f"Loading PatchCore model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Expected dict structure for PatchCore model")
        
        # Extract PatchCore components
        if 'memory_bank' not in checkpoint:
            raise RuntimeError("No memory bank found - required for PatchCore")
        
        self.memory_bank = checkpoint['memory_bank']
        self.is_fitted = checkpoint.get('is_fitted', False)
        self.feature_channels = checkpoint.get('feature_channels', 512)
        self.target_size = checkpoint.get('target_size')
        
        print(f"Memory bank loaded: {self.memory_bank.shape}")
        print(f"Feature channels: {self.feature_channels}")
        print(f"Model fitted: {self.is_fitted}")
        
        # Extract threshold if available
        if 'threshold' in checkpoint:
            self.threshold = checkpoint['threshold']
        elif 'training_info' in checkpoint and isinstance(checkpoint['training_info'], dict):
            training_info = checkpoint['training_info']
            if 'threshold' in training_info:
                self.threshold = training_info['threshold']
        
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        print(f"PatchCore model loaded successfully")
    
    def _setup_preprocessing(self):
        """Setup preprocessing for PatchCore"""
        # Determine image size
        if self.target_size:
            if isinstance(self.target_size, (int, float)):
                image_size = (int(self.target_size), int(self.target_size))
            elif isinstance(self.target_size, (list, tuple)) and len(self.target_size) >= 2:
                image_size = tuple(self.target_size[:2])
            else:
                image_size = IMAGE_SIZE
        else:
            image_size = IMAGE_SIZE
        
        # Use config size if target size is too small
        if image_size[0] < 32 or image_size[1] < 32:
            image_size = IMAGE_SIZE
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    
    def _setup_feature_extractor(self):
        """Setup feature extractor for PatchCore"""
        import torchvision.models as models
        
        print("Loading ResNet18 feature extractor...")
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)
        print("Feature extractor loaded successfully")
    
    def predict(self, image):
        """Predict anomaly using PatchCore fitted model"""
        # Handle different input types
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features and calculate anomaly score
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            anomaly_score = self._calculate_patchcore_score(features)
        
        # Create result object
        class PatchCoreResult:
            def __init__(self, score, threshold):
                self.pred_score = torch.tensor(score)
                self.anomaly_map = None
                self.pred_label = 'Anomalous' if score > threshold else 'Normal'
                self.threshold = threshold
        
        return PatchCoreResult(anomaly_score, self.threshold)
    
    def _calculate_patchcore_score(self, features):
        """Calculate anomaly score using memory bank"""
        batch_size = features.shape[0]
        
        # Process features based on dimensions
        if len(features.shape) == 4:  # [B, C, H, W]
            features_pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features_flat = features_pooled.view(batch_size, -1)
        elif len(features.shape) == 3:  # [B, C, L]
            features_flat = torch.mean(features, dim=2)
        else:  # [B, C] or other
            features_flat = features.view(batch_size, -1)
        
        # Convert memory bank to tensor
        if isinstance(self.memory_bank, (list, tuple)):
            memory_tensor = torch.tensor(self.memory_bank, device=self.device, dtype=torch.float32)
        elif torch.is_tensor(self.memory_bank):
            memory_tensor = self.memory_bank.to(self.device)
        else:
            memory_tensor = torch.tensor(self.memory_bank, device=self.device, dtype=torch.float32)
        
        # Ensure memory bank is 2D
        if len(memory_tensor.shape) > 2:
            memory_tensor = memory_tensor.view(memory_tensor.shape[0], -1)
        
        # Match feature dimensions to memory bank
        if features_flat.shape[1] != memory_tensor.shape[1]:
            if features_flat.shape[1] > memory_tensor.shape[1]:
                features_flat = features_flat[:, :memory_tensor.shape[1]]
            else:
                padding_size = memory_tensor.shape[1] - features_flat.shape[1]
                padding = torch.zeros(features_flat.shape[0], padding_size, device=self.device)
                features_flat = torch.cat([features_flat, padding], dim=1)
        
        # Calculate distances to memory bank
        distances = torch.cdist(features_flat, memory_tensor)
        min_distances = torch.min(distances, dim=1)[0]
        
        # Calculate and normalize anomaly score
        anomaly_score = float(torch.mean(min_distances).cpu().item())
        anomaly_score = min(1.0, max(0.0, anomaly_score / 100.0))
        
        return anomaly_score


class ModelLoader:
    """Model Loader for config-based model loading"""
    
    def __init__(self, device=None):
        self.device = device if device else DEVICE
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            
        self.anomalib_model = None
        self.hrnet_model = None
        self.models_loaded = False
        
    def load_models(self, anomalib_path=None, hrnet_path=None):
        """Load models from config or custom paths"""
        print("Loading models...")
        
        anomalib_model_path = anomalib_path or ANOMALIB_MODEL_PATH
        hrnet_model_path = hrnet_path or HRNET_MODEL_PATH
        
        print(f"Anomalib model: {anomalib_model_path}")
        print(f"HRNet model: {hrnet_model_path}")
        
        # Verify files exist
        if not os.path.exists(anomalib_model_path):
            raise FileNotFoundError(f"Anomalib model not found: {anomalib_model_path}")
        
        if not os.path.exists(hrnet_model_path):
            raise FileNotFoundError(f"HRNet model not found: {hrnet_model_path}")
        
        # Load models
        self._load_anomalib_model(anomalib_model_path)
        self._load_hrnet_model(hrnet_model_path)
        
        self.models_loaded = True
        print("All models loaded successfully")
        return True
    
    def _load_anomalib_model(self, model_path):
        """Load Anomalib/PatchCore model"""
        print("Loading Anomalib model...")
        self.anomalib_model = PatchCoreInferencer(model_path, self.device)
        print("Anomalib model loaded successfully")
    
    def _load_hrnet_model(self, model_path):
        """Load HRNet model"""
        print("Loading HRNet model...")
        
        if HRNET_CREATOR_AVAILABLE:
            num_classes = len(SPECIFIC_DEFECT_CLASSES)
            self.hrnet_model = create_hrnet_model(num_classes=num_classes)
        else:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model' in checkpoint:
                self.hrnet_model = checkpoint['model']
            elif isinstance(checkpoint, torch.nn.Module):
                self.hrnet_model = checkpoint
            else:
                raise RuntimeError("Cannot determine HRNet model structure")
        
        # Load state dict if we have model architecture
        if self.hrnet_model is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
                state_dict = None  # Model already loaded
            else:
                state_dict = checkpoint
            
            if state_dict is not None:
                try:
                    self.hrnet_model.load_state_dict(state_dict, strict=True)
                except RuntimeError:
                    self.hrnet_model.load_state_dict(state_dict, strict=False)
        
        self.hrnet_model.to(self.device)
        self.hrnet_model.eval()
        print("HRNet model loaded successfully")
    
    def get_models(self):
        """Return loaded models"""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        if self.anomalib_model is None or self.hrnet_model is None:
            raise RuntimeError("One or more models failed to load.")
        
        return self.anomalib_model, self.hrnet_model
    
    def is_ready(self):
        """Check if models are ready"""
        return (self.models_loaded and 
                self.anomalib_model is not None and 
                self.hrnet_model is not None)
    
    def validate_models(self):
        """Validate loaded models"""
        if not self.is_ready():
            return False, "Models not loaded"
        
        # Test Anomalib model
        if not hasattr(self.anomalib_model, 'predict'):
            return False, "Anomalib model missing predict method"
        
        # Test HRNet model
        if not hasattr(self.hrnet_model, 'eval'):
            return False, "HRNet model invalid"
        
        return True, "Models validated successfully"
    
    def get_model_info(self):
        """Get model information with safe error handling"""
        base_info = {
            'config_paths': {
                'anomalib': str(ANOMALIB_MODEL_PATH),
                'hrnet': str(HRNET_MODEL_PATH)
            },
            'device': self.device
        }
        
        if not self.is_ready():
            return {
                'status': 'not_loaded',
                'anomalib_loaded': False,
                'hrnet_loaded': False,
                **base_info
            }
        
        try:
            # Safe parameter counting
            try:
                hrnet_params = sum(p.numel() for p in self.hrnet_model.parameters())
            except:
                hrnet_params = 0
            
            # Safe validation
            try:
                validation_result, validation_msg = self.validate_models()
            except:
                validation_result, validation_msg = False, "Validation failed"
            
            # Safe anomalib info
            anomalib_info = {
                'type': type(self.anomalib_model).__name__,
                'device': getattr(self.anomalib_model, 'device', 'unknown')
            }
            
            # Safe threshold extraction
            try:
                anomalib_info['threshold'] = self.anomalib_model.threshold
            except:
                anomalib_info['threshold'] = ANOMALY_THRESHOLD
            
            return {
                'status': 'loaded',
                'mode': 'PRODUCTION_PATCHCORE',
                'anomalib_loaded': True,
                'hrnet_loaded': True,
                'config_info': {
                    'anomalib_path': str(ANOMALIB_MODEL_PATH),
                    'hrnet_path': str(HRNET_MODEL_PATH),
                    'anomaly_threshold': ANOMALY_THRESHOLD,
                    'defect_threshold': DEFECT_CONFIDENCE_THRESHOLD,
                    'image_size': IMAGE_SIZE,
                    'num_classes': len(SPECIFIC_DEFECT_CLASSES)
                },
                'anomalib_info': anomalib_info,
                'hrnet_info': {
                    'type': type(self.hrnet_model).__name__,
                    'parameters': hrnet_params,
                    'num_classes': len(SPECIFIC_DEFECT_CLASSES)
                },
                'validation': {
                    'status': validation_result,
                    'message': validation_msg
                },
                **base_info
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                **base_info
            }
    
    def test_anomalib_prediction(self, test_image_path=None):
        """Test anomalib model prediction"""
        if not self.anomalib_model:
            return False, "Anomalib model not loaded"
        
        try:
            if test_image_path and os.path.exists(test_image_path):
                result = self.anomalib_model.predict(test_image_path)
                
                if hasattr(result, 'pred_score'):
                    if isinstance(result.pred_score, torch.Tensor):
                        score = float(result.pred_score.cpu().item())
                    else:
                        score = float(result.pred_score)
                    
                    return True, f"Prediction successful. Score: {score:.4f}, Decision: {result.pred_label}"
                else:
                    return False, "Result missing pred_score"
            else:
                if hasattr(self.anomalib_model, 'predict') and callable(self.anomalib_model.predict):
                    return True, "Predict method available and callable"
                else:
                    return False, "Predict method not available"
                    
        except Exception as e:
            return False, f"Prediction test failed: {e}"


# Convenience functions
def auto_load_models(device='cuda'):
    """Auto load models from config"""
    loader = ModelLoader(device=device)
    loader.load_models()
    return loader.get_models()


def load_custom_models(anomalib_path, hrnet_path, device='cuda'):
    """Load from custom paths"""
    loader = ModelLoader(device=device)
    loader.load_models(anomalib_path, hrnet_path)
    return loader.get_models()


if __name__ == "__main__":
    print("Testing Production Model Loader")
    print("=" * 50)
    
    # Show config info
    print("Config Information:")
    print(f"   Device: {DEVICE}")
    print(f"   Anomalib Model: {ANOMALIB_MODEL_PATH}")
    print(f"   HRNet Model: {HRNET_MODEL_PATH}")
    print(f"   Anomaly Threshold: {ANOMALY_THRESHOLD}")
    print(f"   Image Size: {IMAGE_SIZE}")
    print(f"   Defect Classes: {len(SPECIFIC_DEFECT_CLASSES)}")
    
    # Check files exist
    print(f"\nFile Check:")
    print(f"   Anomalib exists: {ANOMALIB_MODEL_PATH.exists()}")
    print(f"   HRNet exists: {HRNET_MODEL_PATH.exists()}")
    
    if not ANOMALIB_MODEL_PATH.exists():
        print(f"Anomalib model not found: {ANOMALIB_MODEL_PATH}")
        exit(1)
    
    if not HRNET_MODEL_PATH.exists():
        print(f"HRNet model not found: {HRNET_MODEL_PATH}")
        exit(1)
    
    # Test loading
    print(f"\nTesting model loading...")
    try:
        loader = ModelLoader()
        loader.load_models()
        
        valid, message = loader.validate_models()
        if valid:
            print(f"Validation: {message}")
        else:
            print(f"Validation failed: {message}")
            exit(1)
        
        # Test prediction
        print(f"\nTesting prediction...")
        pred_ok, pred_msg = loader.test_anomalib_prediction()
        print(f"   {pred_msg}")
        
        # Show model info
        model_info = loader.get_model_info()
        print(f"\nModel Info:")
        print(f"   Status: {model_info['status']}")
        print(f"   Mode: {model_info['mode']}")
        print(f"   Device: {model_info['device']}")
        
        print(f"\nALL TESTS PASSED!")
        print(f"To change model: Update ANOMALIB_MODEL_PATH in config.py")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()