# immediate_fix_test.py - Test API dengan format yang benar
"""
Test API dengan cara yang pasti bekerja
Mengatasi masalah Content-Type dan request handling
"""

import requests
import base64
import os

# Configuration
API_URL = "http://localhost:5000"
IMAGE_PATH = "C:/Users/Fitra/Documents/automated_defect/1745296632783_jpg.rf.136d6400d4db0fc531a60042da9f37d3.jpg"

def test_with_base64_method():
    """Test menggunakan base64 method - seharusnya work"""
    print("Testing with Base64 method...")
    
    try:
        # Read and encode image
        with open(IMAGE_PATH, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare JSON payload
        payload = {
            "image_base64": f"data:image/jpeg;base64,{image_data}",
            "filename": os.path.basename(IMAGE_PATH)
        }
        
        # Send request with proper headers
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        print("Sending request...")
        response = requests.post(
            f"{API_URL}/api/detection/image",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print("="*50)
            data = result['data']
            print(f"Analysis ID: {data['analysis_id']}")
            print(f"Decision: {data['final_decision']}")
            print(f"Processing Time: {data['processing_time']:.3f}s")
            print(f"Anomaly Score: {data['anomaly_detection']['anomaly_score']:.4f}")
            print(f"Detected Defects: {data['detected_defects']}")
            print(f"Confidence: {data['confidence_level']}")
            return True
        else:
            print("FAILED!")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_with_requests_files():
    """Test dengan requests files method"""
    print("\nTesting with requests files method...")
    
    try:
        # Open file and prepare for upload
        with open(IMAGE_PATH, 'rb') as f:
            files = {
                'image': (os.path.basename(IMAGE_PATH), f, 'image/jpeg')
            }
            
            print("Sending file upload request...")
            response = requests.post(
                f"{API_URL}/api/detection/image",
                files=files,
                timeout=30
            )
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS with files method!")
            data = result['data']
            print(f"Decision: {data['final_decision']}")
            print(f"Processing Time: {data['processing_time']:.3f}s")
            return True
        else:
            print("Failed with files method")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Files method error: {e}")
        return False

def debug_server_endpoints():
    """Debug what endpoints are available"""
    print("\nDebugging server endpoints...")
    
    # Test health
    try:
        response = requests.get(f"{API_URL}/api/health")
        print(f"Health check: {response.status_code}")
    except:
        print("Health check failed")
    
    # Test system info
    try:
        response = requests.get(f"{API_URL}/api/system/info")
        print(f"System info: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  System ready: {data.get('data', {}).get('system_ready')}")
            print(f"  Models loaded: {data.get('data', {}).get('models_loaded')}")
    except Exception as e:
        print(f"System info failed: {e}")

def test_direct_curl_equivalent():
    """Test yang setara dengan curl command"""
    print("\nTesting curl equivalent...")
    
    try:
        # This is exactly what curl -F does
        with open(IMAGE_PATH, 'rb') as f:
            # Create multipart form data manually
            files = {'image': f}
            
            response = requests.post(
                f"{API_URL}/api/detection/image",
                files=files
            )
        
        print(f"Curl equivalent status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response: {response.text}")
            print(f"Request headers sent: {response.request.headers}")
        else:
            print("SUCCESS with curl equivalent!")
            result = response.json()
            print(f"Decision: {result['data']['final_decision']}")
        
    except Exception as e:
        print(f"Curl equivalent error: {e}")

def main():
    print("IMMEDIATE FIX TEST")
    print("="*50)
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        return
    
    print(f"Image found: {os.path.basename(IMAGE_PATH)}")
    print(f"File size: {os.path.getsize(IMAGE_PATH)} bytes")
    
    # Debug server first
    debug_server_endpoints()
    
    # Try base64 method (most likely to work)
    success1 = test_with_base64_method()
    
    # Try files method
    success2 = test_with_requests_files()
    
    # Try curl equivalent
    test_direct_curl_equivalent()
    
    if success1 or success2:
        print("\n✅ SUCCESS! At least one method worked!")
    else:
        print("\n❌ BOTH METHODS FAILED")
        print("\nPossible issues:")
        print("1. API server endpoint expects different format")
        print("2. Server-side error in request handling")
        print("3. Model loading issue")
        print("\nCheck server console for detailed error messages")

if __name__ == "__main__":
    main()