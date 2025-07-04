#!/usr/bin/env python3
"""
Simple test for dry-run file logging feature.
This script tests the dry-run logger utility directly.
"""

import sys
import os
import json
import glob
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), './llm_factory'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), './llm_factory/utils'))

# Import the dry-run logger directly
try:
    from llm_factory.utils.dry_run_logger import DryRunLogger, create_dry_run_logger
except ImportError:
    # Direct import without going through llm_factory package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), './llm_factory/utils'))
    from dry_run_logger import DryRunLogger, create_dry_run_logger


def cleanup_dry_run_folder():
    """Clean up any existing dry-run files from previous tests"""
    dry_run_folder = "llm_factory_dry_run_responses"
    if os.path.exists(dry_run_folder):
        files = glob.glob(os.path.join(dry_run_folder, "*.json"))
        for file in files:
            try:
                os.remove(file)
            except:
                pass
        print(f"🧹 Cleaned up {len(files)} existing dry-run files")


def test_dry_run_logger_creation():
    """Test that the dry-run logger can be created and initializes properly"""
    print("\n" + "="*50)
    print("Testing Dry-Run Logger Creation")
    print("="*50)
    
    try:
        logger = create_dry_run_logger()
        
        # Check that folder was created
        if os.path.exists(logger.dry_run_folder):
            print(f"✅ Dry-run folder created: {logger.dry_run_folder}")
        else:
            print(f"❌ Dry-run folder not created")
            return False
        
        # Check session info
        session_info = logger.get_session_info()
        required_fields = ['session_id', 'dry_run_folder', 'requests_logged', 'folder_exists']
        
        for field in required_fields:
            if field in session_info:
                print(f"✅ Session info has {field}: {session_info[field]}")
            else:
                print(f"❌ Session info missing {field}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create dry-run logger: {e}")
        return False


def test_request_logging():
    """Test logging individual requests"""
    print("\n" + "="*50)
    print("Testing Request Logging")
    print("="*50)
    
    try:
        logger = create_dry_run_logger()
        
        # Test request data
        request_data = {
            "model": "gpt-4-test",
            "temperature": 0.7,
            "max_tokens": 500,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test message for dry-run logging."}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "test": {"type": "string"}
                    }
                }
            }
        }
        
        metadata = {
            "exclude_from_history": False,
            "exclude_context": True,
            "step_index": 1,
            "fine_tuned": False
        }
        
        # Log a request
        file_path = logger.log_request(
            pipeline_name="test_pipeline",
            step_name="test_step_1",
            step_type="initialPrompt",
            request_data=request_data,
            metadata=metadata
        )
        
        if file_path and os.path.exists(file_path):
            print(f"✅ Request logged to: {os.path.basename(file_path)}")
            
            # Verify file contents
            with open(file_path, 'r', encoding='utf-8') as f:
                logged_data = json.load(f)
            
            # Check required structure
            required_sections = ['session_info', 'pipeline_info', 'request_data', 'metadata']
            for section in required_sections:
                if section in logged_data:
                    print(f"✅ File contains {section}")
                else:
                    print(f"❌ File missing {section}")
                    return False
            
            # Verify specific data
            if logged_data['request_data']['model'] == 'gpt-4-test':
                print("✅ Request data correctly saved")
            else:
                print("❌ Request data not correctly saved")
                return False
            
            return True
        else:
            print(f"❌ Request not logged or file not created")
            return False
        
    except Exception as e:
        print(f"❌ Request logging test failed: {e}")
        return False


def test_pipeline_summary_logging():
    """Test logging pipeline summaries"""
    print("\n" + "="*50)
    print("Testing Pipeline Summary Logging")
    print("="*50)
    
    try:
        logger = create_dry_run_logger()
        
        # Test pipeline config
        pipeline_config = {
            "name": "test_summary_pipeline",
            "model": "claude-3-test",
            "temperature": 0.5,
            "dry_run": True,
            "steps": [
                {"type": "initialPrompt", "name": "step1"},
                {"type": "finalAnswer", "name": "step2"}
            ]
        }
        
        execution_metadata = {
            "fine_tuning_enabled": False,
            "context_data_provided": True,
            "total_failed_steps": 0,
            "session_info": logger.get_session_info()
        }
        
        # Log pipeline summary
        file_path = logger.log_pipeline_summary(
            pipeline_name="test_summary_pipeline",
            total_steps=2,
            pipeline_config=pipeline_config,
            execution_metadata=execution_metadata
        )
        
        if file_path and os.path.exists(file_path):
            print(f"✅ Pipeline summary logged to: {os.path.basename(file_path)}")
            
            # Verify it's a summary file
            if "SUMMARY" in os.path.basename(file_path):
                print("✅ Summary file has correct naming convention")
            else:
                print("❌ Summary file naming incorrect")
                return False
            
            # Verify file contents
            with open(file_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            if summary_data.get('session_info', {}).get('summary_type') == 'pipeline_execution':
                print("✅ Summary file correctly marked as pipeline execution")
            else:
                print("❌ Summary file not correctly marked")
                return False
            
            return True
        else:
            print(f"❌ Pipeline summary not logged")
            return False
        
    except Exception as e:
        print(f"❌ Pipeline summary logging test failed: {e}")
        return False


def test_multiple_requests():
    """Test logging multiple requests and verify unique naming"""
    print("\n" + "="*50)
    print("Testing Multiple Request Logging")
    print("="*50)
    
    try:
        logger = create_dry_run_logger()
        
        # Log multiple requests
        file_paths = []
        for i in range(3):
            request_data = {
                "model": f"test-model-{i}",
                "temperature": 0.1 * i,
                "messages": [{"role": "user", "content": f"Test message {i}"}]
            }
            
            file_path = logger.log_request(
                pipeline_name="multi_test_pipeline",
                step_name=f"test_step_{i}",
                step_type="testStep",
                request_data=request_data
            )
            
            if file_path:
                file_paths.append(file_path)
        
        if len(file_paths) == 3:
            print(f"✅ All 3 requests logged successfully")
            
            # Verify unique filenames
            filenames = [os.path.basename(path) for path in file_paths]
            if len(set(filenames)) == len(filenames):
                print("✅ All filenames are unique")
            else:
                print("❌ Some filenames are not unique")
                return False
            
            # Verify request counter
            session_info = logger.get_session_info()
            if session_info['requests_logged'] == 3:
                print(f"✅ Request counter correct: {session_info['requests_logged']}")
            else:
                print(f"❌ Request counter incorrect: {session_info['requests_logged']}")
                return False
            
            return True
        else:
            print(f"❌ Expected 3 files, got {len(file_paths)}")
            return False
        
    except Exception as e:
        print(f"❌ Multiple request logging test failed: {e}")
        return False


def inspect_all_files():
    """Inspect all created files"""
    print("\n" + "="*50)
    print("Final File Inspection")
    print("="*50)
    
    dry_run_folder = "llm_factory_dry_run_responses"
    if not os.path.exists(dry_run_folder):
        print("❌ Dry-run folder doesn't exist")
        return
    
    files = glob.glob(os.path.join(dry_run_folder, "*.json"))
    
    print(f"📁 Found {len(files)} files in dry-run folder:")
    
    for file_path in sorted(files):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pipeline_name = data.get("pipeline_info", {}).get("pipeline_name", "unknown")
            step_name = data.get("pipeline_info", {}).get("step_name", "unknown")
            timestamp = data.get("session_info", {}).get("timestamp", "unknown")
            
            print(f"📄 {filename}")
            print(f"   Pipeline: {pipeline_name} | Step: {step_name}")
            print(f"   Timestamp: {timestamp}")
            
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")


def main():
    """Run all dry-run logger tests"""
    print("🚀 Starting Dry-Run Logger Test Suite")
    print("This test verifies the core dry-run logging functionality.\n")
    
    # Clean up before starting
    cleanup_dry_run_folder()
    
    tests = [
        ("Dry-Run Logger Creation", test_dry_run_logger_creation),
        ("Request Logging", test_request_logging),
        ("Pipeline Summary Logging", test_pipeline_summary_logging),
        ("Multiple Request Logging", test_multiple_requests)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Inspect created files
    inspect_all_files()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<35} {status}")
        if not success:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! Dry-run file logging is working correctly.")
        print("✅ Files are being created with proper structure and timestamps.")
        print("✅ Unique naming and session tracking working.")
        
        dry_run_folder = "llm_factory_dry_run_responses"
        if os.path.exists(dry_run_folder):
            print(f"📁 Files saved in: {os.path.abspath(dry_run_folder)}")
        
        return 0
    else:
        print("💥 SOME TESTS FAILED! Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())