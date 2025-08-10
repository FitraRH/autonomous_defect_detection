# simple_api_test.py
"""
Simple API Test - Test basic detection
"""

import requests
import base64
import os
import json

API_URL = "http://localhost:5000"
IMAGE_PATH = "C:/Users/Fitra/Documents/automated_defect/1745296632783_jpg.rf.136d6400d4db0fc531a60042da9f37d3.jpg"

def test_health():
    """Test health endpoint"""
    print("🔍 TESTING HEALTH ENDPOINT")
    print("=" * 40)
    
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Health test failed: {e}")
        return False

def test_system_info():
    """Test system info endpoint"""
    print("\n🔍 TESTING SYSTEM INFO")
    print("=" * 40)
    
    try:
        response = requests.get(f"{API_URL}/api/system/info", timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response structure: {list(data.keys())}")
            
            if 'data' in data:
                system_data = data['data']
                print(f"System data keys: {list(system_data.keys())}")
                print(f"Models loaded: {system_data.get('models_loaded')}")
                print(f"System ready: {system_data.get('system_ready')}")
                return system_data.get('system_ready', False)
            else:
                print(f"No 'data' key in response: {data}")
                return False
        else:
            print(f"Error response: {response.text}")
            return False
            
    except Exception as e:
        print(f"System info test failed: {e}")
        return False

def test_detection():
    """Test detection endpoint"""
    print("\n🔍 TESTING DETECTION")
    print("=" * 40)
    
    if not os.path.exists(IMAGE_PATH):
        print("❌ Test image not found")
        return False
    
    try:
        # Read and encode image
        with open(IMAGE_PATH, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare JSON payload
        payload = {
            "image_base64": image_data,
            "filename": os.path.basename(IMAGE_PATH)
        }
        
        print(f"Sending image: {os.path.basename(IMAGE_PATH)}")
        print(f"Image size: {len(image_data)} characters (base64)")
        
        # Send request
        response = requests.post(
            f"{API_URL}/api/detection/image",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Detection successful!")
            
            # Parse result
            if 'data' in result:
                data = result['data']
                print(f"Decision: {data.get('final_decision')}")
                print(f"Processing time: {data.get('processing_time')}s")
                
                if 'anomaly_detection' in data:
                    anomaly = data['anomaly_detection']
                    print(f"Anomaly score: {anomaly.get('anomaly_score')}")
                    print(f"Anomaly decision: {anomaly.get('decision')}")
                
                return True
            else:
                print(f"Unexpected response format: {list(result.keys())}")
                return False
        else:
            print(f"❌ Detection failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data.get('error')}")
            except:
                print(f"Raw error: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def main():
    print("🧪 SIMPLE API TEST")
    print("Testing basic PatchCore detection")
    print("=" * 50)
    
    # Test 1: Health
    health_ok = test_health()
    if not health_ok:
        print("\n❌ Health test failed - check if server is running")
        return
    
    # Test 2: System Info
    system_ok = test_system_info()
    if not system_ok:
        print("\n❌ System not ready - check model loading")
        return
    
    # Test 3: Detection
    detection_ok = test_detection()
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Health: {'✅' if health_ok else '❌'}")
    print(f"System: {'✅' if system_ok else '❌'}")
    print(f"Detection: {'✅' if detection_ok else '❌'}")
    
    if health_ok and system_ok and detection_ok:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"Your PatchCore API is working perfectly!")
    else:
        print(f"\n⚠️ Some tests failed")
        print(f"Check server console for detailed errors")

if __name__ == "__main__":
    main()