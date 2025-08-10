# api_diagnosis.py - COMPREHENSIVE API DIAGNOSIS
"""
API Diagnosis Script
Tests your PatchCore API comprehensively
"""

import requests
import base64
import os
import json
import time
import traceback

# Configuration
API_URL = "http://localhost:5000"
IMAGE_PATH = "C:/Users/Fitra/Documents/automated_defect/1745296632783_jpg.rf.136d6400d4db0fc531a60042da9f37d3.jpg"

def diagnose_server_health():
    """Test server connectivity and basic health"""
    print("🔍 TESTING SERVER HEALTH")
    print("=" * 50)
    
    try:
        # Test basic connectivity
        print("📡 Testing server connectivity...")
        response = requests.get(f"{API_URL}/api/health", timeout=10)
        print(f"✅ Server reachable: {response.status_code}")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health status: {health_data.get('status')}")
            print(f"📊 API version: {health_data.get('api_version', 'N/A')}")
            
            # Show services status
            services = health_data.get('services', {})
            if services:
                print("🔧 Services status:")
                for service, status in services.items():
                    status_icon = "✅" if status.get('ready', False) else "❌"
                    print(f"   {status_icon} {service}: {status.get('status', 'unknown')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test system info
        print("\n📋 Testing system info...")
        response = requests.get(f"{API_URL}/api/system/info", timeout=15)
        if response.status_code == 200:
            system_data = response.json().get('data', {})
            print(f"✅ System info retrieved")
            print(f"   Models loaded: {system_data.get('models_loaded', 'Unknown')}")
            print(f"   System ready: {system_data.get('system_ready', 'Unknown')}")
            print(f"   Device: {system_data.get('device', 'Unknown')}")
            print(f"   Detector ready: {system_data.get('detector_ready', 'Unknown')}")
            
            # Check if system is ready for detection
            if not system_data.get('system_ready', False):
                print("❌ CRITICAL: System not ready for detection")
                return False
            
            # Show model info if available
            if 'anomalib_info' in system_data:
                anomalib_info = system_data['anomalib_info']
                print(f"   🎯 Model type: {anomalib_info.get('model_type', 'Unknown')}")
                print(f"   🎯 Is PatchCore: {anomalib_info.get('is_patchcore', 'Unknown')}")
                print(f"   🎯 Has Memory Bank: {anomalib_info.get('has_memory_bank', 'Unknown')}")
        else:
            print(f"❌ System info failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ CRITICAL: Cannot connect to server")
        print("   💡 Is the server running? Try: python api_server.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ CRITICAL: Server timeout")
        print("   💡 Server may be overloaded or models loading")
        return False
    except Exception as e:
        print(f"❌ CRITICAL: Health check error: {e}")
        return False

def test_image_detection():
    """Test image detection with comprehensive analysis"""
    print("\n🔍 TESTING IMAGE DETECTION")
    print("=" * 50)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Test image not found: {IMAGE_PATH}")
        print("💡 Update IMAGE_PATH in script to point to your test image")
        return False
    
    try:
        # Load and prepare image
        print(f"📸 Loading test image: {os.path.basename(IMAGE_PATH)}")
        with open(IMAGE_PATH, 'rb') as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        print(f"✅ Image loaded: {len(image_data)} bytes, {len(image_base64)} base64 chars")
        
        # Test JSON detection
        print("\n🧪 Testing JSON detection...")
        payload = {
            "image_base64": image_base64,
            "filename": os.path.basename(IMAGE_PATH)
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/api/detection/image",
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        processing_time = time.time() - start_time
        
        print(f"📊 Response time: {processing_time:.2f} seconds")
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response size: {len(response.content)} bytes")
        
        if response.status_code == 200:
            print("🎉 DETECTION SUCCESS!")
            
            try:
                result = response.json()
                data = result.get('data', {})
                
                # Show detection results
                print(f"\n📊 DETECTION RESULTS:")
                print(f"   🎯 Final Decision: {data.get('final_decision', 'Unknown')}")
                print(f"   📊 Processing Time: {data.get('processing_time', 'Unknown')} seconds")
                print(f"   🔍 Analysis ID: {data.get('analysis_id', 'None')}")
                
                # Anomaly detection details
                anomaly_detection = data.get('anomaly_detection', {})
                if anomaly_detection:
                    print(f"\n🔬 ANOMALY DETECTION:")
                    print(f"   📊 Anomaly Score: {anomaly_detection.get('anomaly_score', 'Unknown')}")
                    print(f"   🎯 Decision: {anomaly_detection.get('decision', 'Unknown')}")
                    print(f"   📏 Threshold: {anomaly_detection.get('threshold_used', 'Unknown')}")
                
                # Defect detection details
                detected_defects = data.get('detected_defects', [])
                defect_count = data.get('defect_count', 0)
                
                print(f"\n🔍 DEFECT ANALYSIS:")
                print(f"   📊 Defect Count: {defect_count}")
                if detected_defects:
                    print(f"   🎯 Detected Defects: {', '.join(detected_defects)}")
                else:
                    print(f"   ✅ No specific defects detected")
                
                # Confidence and summary
                confidence = data.get('confidence_level', 'Unknown')
                print(f"\n📈 QUALITY METRICS:")
                print(f"   🎯 Confidence Level: {confidence}")
                
                # Result summary
                result_summary = data.get('result_summary', {})
                if result_summary:
                    print(f"   📊 Is Defective: {result_summary.get('is_defective', 'Unknown')}")
                    print(f"   📊 Processing Status: {result_summary.get('processing_status', 'Unknown')}")
                
                return True
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"Raw response: {response.text[:500]}...")
                return False
                
        else:
            print(f"❌ DETECTION FAILED: {response.status_code}")
            try:
                error_data = response.json()
                error_msg = error_data.get('error', 'Unknown error')
                print(f"Error message: {error_msg}")
                
                # Debug info if available
                debug_info = error_data.get('debug_info', {})
                if debug_info:
                    print(f"Debug info:")
                    for key, value in debug_info.items():
                        print(f"   {key}: {value}")
                        
            except:
                print(f"Raw error response: {response.text[:500]}...")
            
            return False
        
    except requests.exceptions.Timeout:
        print("❌ Detection timeout (>60s)")
        print("💡 Model may be processing slowly or stuck")
        return False
    except Exception as e:
        print(f"❌ Detection error: {e}")
        traceback.print_exc()
        return False

def test_multipart_detection():
    """Test multipart file upload detection"""
    print("\n🔍 TESTING MULTIPART DETECTION")
    print("=" * 50)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Test image not found")
        return False
    
    try:
        print("📤 Testing multipart/form-data upload...")
        
        with open(IMAGE_PATH, 'rb') as f:
            files = {'image': (os.path.basename(IMAGE_PATH), f, 'image/jpeg')}
            
            start_time = time.time()
            response = requests.post(
                f"{API_URL}/api/detection/image",
                files=files,
                timeout=60
            )
            processing_time = time.time() - start_time
        
        print(f"📊 Multipart response time: {processing_time:.2f} seconds")
        print(f"📊 Multipart response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Multipart detection successful!")
            result = response.json()
            data = result.get('data', {})
            print(f"   🎯 Decision: {data.get('final_decision', 'Unknown')}")
            return True
        else:
            print(f"❌ Multipart detection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Multipart test error: {e}")
        return False

def test_batch_detection():
    """Test batch detection capability"""
    print("\n🔍 TESTING BATCH DETECTION")
    print("=" * 50)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Test image not found")
        return False
    
    try:
        print("📦 Testing batch detection with 2 copies of test image...")
        
        # Load image
        with open(IMAGE_PATH, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create batch payload
        batch_payload = {
            "images": [
                {
                    "image_base64": image_base64,
                    "filename": f"test_image_1_{os.path.basename(IMAGE_PATH)}"
                },
                {
                    "image_base64": image_base64,
                    "filename": f"test_image_2_{os.path.basename(IMAGE_PATH)}"
                }
            ]
        }
        
        start_time = time.time()
        response = requests.post(
            f"{API_URL}/api/detection/batch",
            json=batch_payload,
            headers={'Content-Type': 'application/json'},
            timeout=120
        )
        processing_time = time.time() - start_time
        
        print(f"📊 Batch response time: {processing_time:.2f} seconds")
        print(f"📊 Batch response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Batch detection successful!")
            result = response.json()
            data = result.get('data', {})
            
            # Show batch summary
            summary = data.get('summary', {})
            if summary:
                print(f"   📊 Total images: {summary.get('total_images', 'Unknown')}")
                print(f"   ✅ Good products: {summary.get('good_products', 'Unknown')}")
                print(f"   ❌ Defective products: {summary.get('defective_products', 'Unknown')}")
                print(f"   📈 Defect rate: {summary.get('defect_rate', 'Unknown')}%")
            
            return True
        else:
            print(f"❌ Batch detection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Batch test error: {e}")
        return False

def test_performance_metrics():
    """Test performance monitoring endpoints"""
    print("\n🔍 TESTING PERFORMANCE METRICS")
    print("=" * 50)
    
    try:
        print("📊 Testing performance metrics endpoint...")
        response = requests.get(f"{API_URL}/api/performance/metrics", timeout=15)
        
        if response.status_code == 200:
            print("✅ Performance metrics retrieved")
            data = response.json().get('data', {})
            
            # Show performance overview
            overview = data.get('performance_overview', {})
            if overview:
                print(f"   📊 System metrics available: {len(overview)} metrics")
            
            # Show system metrics
            system_metrics = data.get('system_metrics', {})
            if system_metrics:
                cpu = system_metrics.get('cpu', {})
                memory = system_metrics.get('memory', {})
                print(f"   💻 CPU usage: {cpu.get('usage_percent', 'Unknown')}%")
                print(f"   🧠 Memory usage: {memory.get('usage_percent', 'Unknown')}%")
            
            return True
        else:
            print(f"❌ Performance metrics failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def comprehensive_diagnosis():
    """Run comprehensive API diagnosis"""
    print("🧪 COMPREHENSIVE API DIAGNOSIS")
    print("Testing your PatchCore detection API")
    print("=" * 60)
    
    # Test results tracking
    test_results = {}
    
    # Test 1: Server Health
    test_results['server_health'] = diagnose_server_health()
    
    if not test_results['server_health']:
        print("\n❌ CRITICAL: Server health check failed")
        print("🛠️ Fix server issues before testing detection")
        return test_results
    
    # Test 2: Image Detection (Main test)
    test_results['image_detection'] = test_image_detection()
    
    # Test 3: Multipart Detection
    test_results['multipart_detection'] = test_multipart_detection()
    
    # Test 4: Batch Detection
    test_results['batch_detection'] = test_batch_detection()
    
    # Test 5: Performance Metrics
    test_results['performance_metrics'] = test_performance_metrics()
    
    return test_results

def show_diagnosis_summary(test_results):
    """Show comprehensive diagnosis summary"""
    print("\n" + "=" * 60)
    print("📊 DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    # Count successes
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    
    print(f"📈 Overall Success Rate: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
    print("\n📋 Test Results:")
    
    for test_name, result in test_results.items():
        status_icon = "✅" if result else "❌"
        test_display = test_name.replace('_', ' ').title()
        print(f"   {status_icon} {test_display}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if passed_tests == total_tests:
        print("🎉 EXCELLENT: All tests passed! Your API is working perfectly!")
        print("💎 Your PatchCore detection system is production-ready!")
    elif passed_tests >= total_tests * 0.8:
        print("✅ GOOD: Most tests passed! Minor issues detected.")
        print("🔧 Check failed tests for optimization opportunities.")
    elif passed_tests >= total_tests * 0.6:
        print("⚠️ FAIR: Some tests failed. System partially working.")
        print("🔧 Address failed tests to improve reliability.")
    else:
        print("❌ POOR: Many tests failed. Significant issues detected.")
        print("🛠️ Major troubleshooting required.")
    
    # Specific recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if not test_results.get('server_health'):
        print("   🚨 Priority 1: Fix server health issues")
    elif not test_results.get('image_detection'):
        print("   🚨 Priority 1: Fix core image detection")
    else:
        print("   ✅ Core functionality working!")
        
        if not test_results.get('multipart_detection'):
            print("   🔧 Consider: Fix multipart upload support")
        if not test_results.get('batch_detection'):
            print("   🔧 Consider: Fix batch processing support")
        if not test_results.get('performance_metrics'):
            print("   🔧 Consider: Fix performance monitoring")

def main():
    print("🔧 API DIAGNOSIS TOOL")
    print("Comprehensive testing of your PatchCore Detection API")
    print(f"🌐 Testing server: {API_URL}")
    print(f"📸 Test image: {os.path.basename(IMAGE_PATH) if os.path.exists(IMAGE_PATH) else 'NOT FOUND'}")
    print("")
    
    # Run comprehensive diagnosis
    test_results = comprehensive_diagnosis()
    
    # Show summary
    show_diagnosis_summary(test_results)
    
    print(f"\n📁 Image file: {os.path.basename(IMAGE_PATH)}")
    if os.path.exists(IMAGE_PATH):
        print(f"📏 Image size: {os.path.getsize(IMAGE_PATH)} bytes")
    else:
        print(f"❌ Image not found - update IMAGE_PATH in script")

if __name__ == "__main__":
    main()