# safe_json_save.py - Safe JSON saving with encoding handling
import json
import os
from datetime import datetime
from pathlib import Path

def safe_save_json(data, filepath, encoding='utf-8'):
    """Safely save JSON data with proper encoding"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert any problematic unicode in the data
        cleaned_data = clean_unicode_recursive(data)
        
        # Save with explicit encoding
        with open(filepath, 'w', encoding=encoding) as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False, default=str)
        
        return True
        
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False

def clean_unicode_recursive(obj):
    """Recursively clean unicode characters from data structures"""
    if isinstance(obj, dict):
        return {key: clean_unicode_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_unicode_recursive(item) for item in obj]
    elif isinstance(obj, str):
        # Replace problematic unicode with safe alternatives
        replacements = {
            '🔍': '[DETECT]', '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARNING]',
            '🎉': '[SUCCESS]', '📊': '[CHART]', '📁': '[FOLDER]', '🖼️': '[IMAGE]',
            '⚡': '[PERF]', '🔧': '[CONFIG]', '🎯': '[TARGET]', '🚀': '[RUN]',
            '💾': '[SAVE]', '📄': '[REPORT]', '🔄': '[PROCESS]'
        }
        
        cleaned = obj
        for unicode_char, replacement in replacements.items():
            cleaned = cleaned.replace(unicode_char, replacement)
        
        return cleaned
    else:
        return obj

def safe_save_text_report(content, filepath, encoding='utf-8'):
    """Safely save text report with encoding handling"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Clean unicode characters
        cleaned_content = clean_unicode_recursive(content)
        
        # Save with explicit encoding
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(cleaned_content)
        
        return True
        
    except Exception as e:
        print(f"Error saving text report: {e}")
        return False

if __name__ == "__main__":
    # Test the safe save functions
    test_data = {
        'status': '✅ Success',
        'message': '🎉 Detection completed!',
        'results': ['📊 Good', '❌ Defect']
    }
    
    success = safe_save_json(test_data, 'test_safe_save.json')
    print(f"Safe JSON save test: {'✓ Success' if success else '✗ Failed'}")
