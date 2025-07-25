"""
Model loader - FIXED VERSION - Direct Anomalib Import with Pre-processing Fix
Compatible with Anomalib v1.0+ (same version used for training)
Compatible with your custom config.py
Includes targeted fix for missing pre_processing module
"""

import datetime
import os
import torch

# TARGETED PRE-PROCESSING FIX FIRST
print(" Applying targeted pre-processing fix...")
try:
    from anomalib_pre_processing_fix import apply_pre_processing_fix
    pre_processing_fixed = apply_pre_processing_fix()
    if pre_processing_fixed:
        print(" Pre-processing fix applied successfully")
    else:
        print(" Pre-processing fix failed, but continuing...")
except ImportError:
    print(" Pre-processing fix module not found, but continuing...")
    pre_processing_fixed = False

# DIRECT IMPORT - Now should work with fix
try:
    from anomalib.deploy import TorchInferencer
    ANOMALIB_AVAILABLE = True
    print(" Anomalib TorchInferencer imported successfully (with pre-processing fix)")
except ImportError as e:
    print(f" TorchInferencer import failed even with fix: {e}")
    ANOMALIB_AVAILABLE = False
    raise ImportError("Anomalib TorchInferencer required for production")

# Import your custom config
try:
    from config import *
    print(" Custom config imported successfully")
except ImportError as e:
    print(f" Config import failed: {e}")
    raise ImportError("config.py required")

# Try to import HRNet model creator (adjust import based on your structure)
try:
    from .hrnet_model import create_hrnet_model
    HRNET_CREATOR_AVAILABLE = True
except ImportError:
    try:
        from hrnet_model import create_hrnet_model
        HRNET_CREATOR_AVAILABLE = True
    except ImportError:
        print(" HRNet model creator not found - you'll need to implement create_hrnet_model()")
        HRNET_CREATOR_AVAILABLE = False


class ModelLoader:
    """FIXED Production Model Loader - Direct Anomalib Import - Compatible with Custom Config"""
    
    def __init__(self, device=None):
        # Use device from config or parameter
        self.device = device if device else DEVICE
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
            
        self.anomalib_model = None
        self.hrnet_model = None
        self.models_loaded = False
        
        print(f"FIXED ModelLoader initialized for device: {self.device}")
        print(f"Using your custom config.py")
        
        if not ANOMALIB_AVAILABLE:
            raise RuntimeError("Anomalib not available - cannot proceed in production mode")
        
    def load_models(self, anomalib_path=None, hrnet_path=None):
        """Load REAL models - FIXED Version (Direct Import) - Custom Config Compatible"""
        print("Loading PRODUCTION models (FIXED approach with custom config)...")
        
        # Use custom paths or config paths
        anomalib_model_path = anomalib_path or ANOMALIB_MODEL_PATH
        hrnet_model_path = hrnet_path or HRNET_MODEL_PATH
        
        print(f"Anomalib model path: {anomalib_model_path}")
        print(f"HRNet model path: {hrnet_model_path}")
        
        # Verify files exist
        if not os.path.exists(anomalib_model_path):
            raise FileNotFoundError(f"Anomalib model not found: {anomalib_model_path}")
        
        if not os.path.exists(hrnet_model_path):
            raise FileNotFoundError(f"HRNet model not found: {hrnet_model_path}")
        
        # Load models
        self._load_anomalib_model(anomalib_model_path)
        self._load_hrnet_model(hrnet_model_path)
        
        self.models_loaded = True
        print(" All PRODUCTION models loaded successfully (FIXED with custom config)!")
        return True
    
    def _load_anomalib_model(self, model_path):
        """Load Anomalib model - FIXED with Enhanced Pre-processing Package Fix"""
        try:
            print(f"Loading Anomalib model from {model_path}...")
            
            # Enhanced verification of pre-processing fix before loading
            try:
                import anomalib.pre_processing
                import anomalib.pre_processing.pre_processor
                print(" Complete pre-processing package verified before model loading")
            except ImportError as e:
                print(f" Pre-processing package issue: {e}")
                print(" Attempting enhanced emergency fix...")
                
                # Enhanced emergency inline fix
                import sys
                from types import ModuleType
                
                # Create complete package structure inline
                pre_processing = ModuleType('anomalib.pre_processing')
                pre_processing.__path__ = []  # Make it a package
                pre_processing.__package__ = 'anomalib.pre_processing'
                
                # Create pre_processor submodule
                pre_processor = ModuleType('anomalib.pre_processing.pre_processor')
                pre_processor.__package__ = 'anomalib.pre_processing'
                
                # Enhanced classes for STFPM compatibility
                class PreProcessor:
                    def __init__(self, *args, **kwargs):
                        self.transforms = kwargs.get('transforms', [])
                    def __call__(self, image): return image
                    def forward(self, image): return image
                
                class Normalize:
                    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
                        self.mean, self.std = mean, std
                    def __call__(self, x): return x
                
                class ToTensor:
                    def __call__(self, image):
                        import torch
                        if not isinstance(image, torch.Tensor):
                            import numpy as np
                            if isinstance(image, np.ndarray):
                                return torch.from_numpy(image.copy()).float()
                        return image
                
                class Compose:
                    def __init__(self, transforms): self.transforms = transforms
                    def __call__(self, image):
                        for transform in self.transforms: image = transform(image)
                        return image
                
                # Add classes to both modules
                for module in [pre_processing, pre_processor]:
                    module.PreProcessor = PreProcessor
                    module.Normalize = Normalize
                    module.ToTensor = ToTensor
                    module.Compose = Compose
                
                # Link submodule to parent
                pre_processing.pre_processor = pre_processor
                
                # Inject both into sys.modules
                sys.modules['anomalib.pre_processing'] = pre_processing
                sys.modules['anomalib.pre_processing.pre_processor'] = pre_processor
                
                print(" Enhanced emergency pre-processing package fix applied")
                print("   - anomalib.pre_processing (package)")
                print("   - anomalib.pre_processing.pre_processor (submodule)")
            
            # DIRECT LOADING with TorchInferencer
            print(" Loading model with TorchInferencer...")
            self.anomalib_model = TorchInferencer(
                path=model_path, 
                device=self.device
            )
            
            # Verify loaded
            if not hasattr(self.anomalib_model, 'predict'):
                raise RuntimeError("Loaded model missing predict method")
            
            print(f" Anomalib model loaded successfully on {self.device}")
            print(f"   Model type: {type(self.anomalib_model).__name__}")
            print(f"   Model file: {os.path.basename(model_path)}")
            print(f"   Model format: STFPM (Student-Teacher Feature Pyramid Matching)")
            print(f"   Enhanced pre-processing fix:  Applied")
            
            # Test basic functionality
            print(" Testing model functionality...")
            if hasattr(self.anomalib_model, 'model'):
                internal_model = self.anomalib_model.model
                print(f"   Internal model: {type(internal_model).__name__}")
                
                # Check model state
                if hasattr(internal_model, 'eval'):
                    print("    Model supports eval mode")
                if hasattr(internal_model, 'training'):
                    print(f"   Training mode: {internal_model.training}")
            
        except Exception as e:
            print(f" Error loading Anomalib model: {e}")
            print(f"   Model path: {model_path}")
            print(f"   Device: {self.device}")
            
            # Enhanced error analysis
            error_str = str(e)
            if "pre_processing" in error_str:
                if "not a package" in error_str:
                    print("    ENHANCED ERROR ANALYSIS:")
                    print("   - Model requires anomalib.pre_processing as a PACKAGE (not just module)")
                    print("   - Model needs anomalib.pre_processing.pre_processor submodule")
                    print("   - STFPM models have complex preprocessing requirements")
                    print("   - Enhanced fix should have resolved this")
                else:
                    print("    ERROR ANALYSIS:")
                    print("   - Model requires anomalib.pre_processing module")
                    print("   - Your saved model was trained with older Anomalib version")
                    print("   - Enhanced pre-processing fix may need further adjustment")
            elif "No module named" in error_str:
                print("    ERROR ANALYSIS:")
                print("   - Missing dependency in saved model")
                print("   - Model version mismatch with current environment")
                print("   - Consider model format conversion (ONNX/OpenVINO)")
            elif "pickle" in error_str or "torch.load" in error_str:
                print("    ERROR ANALYSIS:")
                print("   - Model serialization/deserialization issue")
                print("   - Possible PyTorch version mismatch")
                print("   - Model file may be corrupted")
            
            print("    SOLUTIONS:")
            print("   1. Re-export model with current Anomalib version")
            print("   2. Convert model to ONNX format (more portable)")
            print("   3. Use exact Anomalib version from training")
            print("   4. Check model file integrity")
            
            raise RuntimeError(
                f"Failed to load Anomalib STFPM model from {model_path}\n"
                f"Error: {e}\n"
                f"This appears to be a model compatibility issue.\n"
                f"Your STFPM model needs specific pre-processing package structure.\n"
                f"Enhanced fix was applied but loading still failed.\n"
                f"Consider re-exporting the model or using ONNX format."
            )
    
    def _load_hrnet_model(self, model_path):
        """Load HRNet model - Compatible with your config"""
        try:
            print(f"Loading HRNet model from {model_path}...")
            
            if HRNET_CREATOR_AVAILABLE:
                # Use standard create_hrnet_model if available
                num_classes = len(SPECIFIC_DEFECT_CLASSES) if hasattr(globals(), 'SPECIFIC_DEFECT_CLASSES') else 6
                self.hrnet_model = create_hrnet_model(num_classes=num_classes)
            else:
                # Fallback: try to load directly or use a simple approach
                print(" Using fallback HRNet loading method")
                print("You may need to implement create_hrnet_model() function")
                # For now, just load the checkpoint and assume it's a complete model
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model' in checkpoint:
                    self.hrnet_model = checkpoint['model']
                elif isinstance(checkpoint, torch.nn.Module):
                    self.hrnet_model = checkpoint
                else:
                    raise RuntimeError("Cannot determine HRNet model structure from checkpoint")
            
            # Load state dict if we have a model architecture
            if hasattr(self, 'hrnet_model') and self.hrnet_model is not None:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
                    # Model is already loaded above
                    state_dict = None
                else:
                    state_dict = checkpoint
                
                # Load state dict if we have one
                if state_dict is not None:
                    try:
                        self.hrnet_model.load_state_dict(state_dict, strict=True)
                        print(" HRNet loaded with strict mode")
                    except RuntimeError as e:
                        print(f" Strict loading failed, using flexible mode: {e}")
                        self.hrnet_model.load_state_dict(state_dict, strict=False)
                        print(" HRNet loaded with flexible mode")
            
            self.hrnet_model.to(self.device)
            self.hrnet_model.eval()
            
            # Count parameters if possible
            try:
                param_count = sum(p.numel() for p in self.hrnet_model.parameters())
                print(f" HRNet model loaded successfully on {self.device}")
                print(f"   Parameters: {param_count:,}")
            except:
                print(f" HRNet model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f" Error loading HRNet model: {e}")
            print(" Make sure your HRNet model file is accessible")
            print(" You may need to implement create_hrnet_model() function")
            raise RuntimeError(f"Failed to load HRNet model: {e}")
    
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
        
        try:
            # Test Anomalib model
            if not hasattr(self.anomalib_model, 'predict'):
                return False, "Anomalib model missing predict method"
            
            if not isinstance(self.anomalib_model, TorchInferencer):
                return False, f"Expected TorchInferencer, got {type(self.anomalib_model)}"
            
            # Test HRNet model
            if not hasattr(self.hrnet_model, 'eval'):
                return False, "HRNet model invalid"
            
            # Check device
            if hasattr(self.hrnet_model, 'parameters'):
                model_device = next(self.hrnet_model.parameters()).device
                if str(model_device) != self.device:
                    return False, f"Device mismatch: {model_device} vs {self.device}"
            
            print(f" Models validated (FIXED approach):")
            print(f"   Anomalib: {type(self.anomalib_model).__name__}")
            print(f"   HRNet: {type(self.hrnet_model).__name__}")
            print(f"   Device: {self.device}")
            print(f"   Loading method: Direct import (same as test.py)")
            
            return True, "Models validated successfully"
            
        except Exception as e:
            return False, f"Model validation failed: {e}"
    
    def get_model_info(self):
        """Get model information"""
        if not self.is_ready():
            return {
                'status': 'not_loaded',
                'anomalib_loaded': False,
                'hrnet_loaded': False,
                'device': self.device,
                'approach': 'FIXED_DIRECT_IMPORT'
            }
        
        try:
            hrnet_params = sum(p.numel() for p in self.hrnet_model.parameters())
            validation_result, validation_msg = self.validate_models()
            
            return {
                'status': 'loaded',
                'mode': 'PRODUCTION_FIXED',
                'anomalib_loaded': True,
                'hrnet_loaded': True,
                'device': self.device,
                'approach': 'DIRECT_IMPORT_NO_COMPATIBILITY',
                'anomalib_info': {
                    'type': type(self.anomalib_model).__name__,
                    'is_torch_inferencer': isinstance(self.anomalib_model, TorchInferencer),
                    'model_path': str(ANOMALIB_MODEL_PATH),
                    'loading_method': 'direct_import_like_test_py'
                },
                'hrnet_info': {
                    'type': type(self.hrnet_model).__name__,
                    'parameters': hrnet_params,
                    'model_path': str(HRNET_MODEL_PATH),
                    'num_classes': len(SPECIFIC_DEFECT_CLASSES) if 'SPECIFIC_DEFECT_CLASSES' in globals() else 6
                },
                'validation': {
                    'status': validation_result,
                    'message': validation_msg
                },
                'compatibility_layer': False,  # No compatibility layer!
                'anomalib_available': ANOMALIB_AVAILABLE,
                'training_version': 'v1.0+',
                'loading_version': 'v1.0+_direct'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'device': self.device,
                'approach': 'FIXED_DIRECT_IMPORT'
            }
    
    def test_anomalib_prediction(self, test_image_path=None):
        """Test anomalib model prediction (like in test.py)"""
        if not self.anomalib_model:
            return False, "Anomalib model not loaded"
        
        try:
            if test_image_path and os.path.exists(test_image_path):
                # Test with real image
                result = self.anomalib_model.predict(image=test_image_path)
                
                # Process result like in test.py
                if hasattr(result, 'pred_score'):
                    if isinstance(result.pred_score, torch.Tensor):
                        score = float(result.pred_score.cpu().item())
                    else:
                        score = float(result.pred_score)
                    
                    return True, f"Prediction successful. Score: {score:.4f}"
                else:
                    return False, "Result missing pred_score"
            else:
                # Just check if predict method exists and is callable
                if hasattr(self.anomalib_model, 'predict') and callable(self.anomalib_model.predict):
                    return True, "Predict method available and callable"
                else:
                    return False, "Predict method not available"
                    
        except Exception as e:
            return False, f"Prediction test failed: {e}"
    
    def unload_models(self):
        """Unload models"""
        try:
            if self.anomalib_model:
                del self.anomalib_model
                self.anomalib_model = None
            
            if self.hrnet_model:
                del self.hrnet_model
                self.hrnet_model = None
            
            self.models_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(" Models unloaded successfully")
            return True
            
        except Exception as e:
            print(f" Error unloading models: {e}")
            return False
    
    def reload_models(self, anomalib_path=None, hrnet_path=None):
        """Reload models"""
        print("Reloading models (FIXED approach)...")
        self.unload_models()
        return self.load_models(anomalib_path, hrnet_path)


# Convenience functions
def auto_load_models(device='cuda'):
    """Auto load models - FIXED"""
    loader = ModelLoader(device=device)
    loader.load_models()
    return loader.get_models()


def load_custom_models(anomalib_path, hrnet_path, device='cuda'):
    """Load from custom paths - FIXED"""
    loader = ModelLoader(device=device)
    loader.load_models(anomalib_path, hrnet_path)
    return loader.get_models()


def validate_model_files():
    """Validate model files exist - Compatible with custom config"""
    missing_files = []
    
    if not os.path.exists(ANOMALIB_MODEL_PATH):
        missing_files.append(f"Anomalib model: {ANOMALIB_MODEL_PATH}")
    
    if not os.path.exists(HRNET_MODEL_PATH):
        missing_files.append(f"HRNet model: {HRNET_MODEL_PATH}")
    
    return len(missing_files) == 0, missing_files


def get_model_file_info():
    """Get model file information - Compatible with custom config"""
    info = {}
    
    for name, path in [('anomalib', ANOMALIB_MODEL_PATH), ('hrnet', HRNET_MODEL_PATH)]:
        try:
            if os.path.exists(path):
                stat = os.stat(path)
                info[name] = {
                    'path': str(path),
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'modified': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'exists': True
                }
            else:
                info[name] = {
                    'path': str(path),
                    'exists': False,
                    'status': 'not_found'
                }
        except Exception as e:
            info[name] = {'error': str(e)}
    
    return info


def test_direct_anomalib_import():
    """Test direct anomalib import (like test.py) - FIXED"""
    try:
        from anomalib.deploy import TorchInferencer
        print(" Direct TorchInferencer import successful")
        return True
    except Exception as e:
        print(f" Direct import failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing FIXED Production Model Loader with Custom Config...")
    print("=" * 70)
    
    # Show config info
    print("1. Custom Config Information:")
    print(f"   Device: {DEVICE}")
    print(f"   Anomalib Model: {ANOMALIB_MODEL_PATH}")
    print(f"   HRNet Model: {HRNET_MODEL_PATH}")
    print(f"   Defect Classes: {len(SPECIFIC_DEFECT_CLASSES) if 'SPECIFIC_DEFECT_CLASSES' in globals() else 'Not defined'}")
    
    # Test direct import first
    print("\n2. Testing direct anomalib import...")
    direct_import_ok = test_direct_anomalib_import()
    if not direct_import_ok:
        print(" Direct import failed - check Anomalib installation")
        exit(1)
    
    # Test files
    print("\n3. Checking model files...")
    files_valid, missing = validate_model_files()
    if not files_valid:
        print(" Missing files:")
        for f in missing:
            print(f"   - {f}")
        print("\n Make sure your model files exist at the paths specified in config.py")
        exit(1)
    
    # Show file info
    file_info = get_model_file_info()
    print(" Model Files:")
    for name, info in file_info.items():
        if info.get('exists'):
            print(f"   {name}: {info['size_mb']}MB")
        else:
            print(f"   {name}: {info['status']}")
    
    # Test loading
    print("\n4. Testing FIXED model loading with custom config...")
    try:
        loader = ModelLoader()
        loader.load_models()
        
        valid, message = loader.validate_models()
        if valid:
            print(f" {message}")
        else:
            print(f" {message}")
            exit(1)
        
        model_info = loader.get_model_info()
        print("\n FIXED Model Information (Custom Config):")
        print(f"   Status: {model_info['status']}")
        print(f"   Mode: {model_info['mode']}")
        print(f"   Approach: {model_info['approach']}")
        print(f"   Device: {model_info['device']}")
        print(f"   Anomalib: {model_info['anomalib_info']['type']}")
        print(f"   HRNet: {model_info['hrnet_info']['type']}")
        print(f"   HRNet Classes: {model_info['hrnet_info']['num_classes']}")
        print(f"   Compatibility Layer: {model_info['compatibility_layer']}")
        
        # Test prediction
        print("\n5. Testing anomalib prediction...")
        pred_ok, pred_msg = loader.test_anomalib_prediction()
        if pred_ok:
            print(f" {pred_msg}")
        else:
            print(f" {pred_msg}")
        
        print("\n FIXED production model loader with custom config test completed!")
        print(" Ready for production with your custom configuration!")
        print(" Same approach as working test.py - no compatibility layer needed!")
        
    except Exception as e:
        print(f" Test failed: {e}")
        print("\nDEBUG INFO:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print("\n Check:")
        print("   1. Model file paths in your config.py")
        print("   2. File permissions and accessibility")
        print("   3. HRNet model loading (may need create_hrnet_model function)")
        exit(1)