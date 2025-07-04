#!/usr/bin/env python3
"""
Test script for enhanced dry-run file logging features.
This script tests that all dry-run requests are properly logged to JSON files.
"""

import json
import logging
import sys
import os
import glob
from datetime import datetime

# Add the llm_factory to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import only the specific modules we need to avoid dependency issues
try:
    from llm_factory.processors.cot_processor import ChainOfThoughtProcessor
    from llm_factory.processors.standard_processor import StandardPromptProcessor
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct import...")
    # Add utils to path for relative imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), './llm_factory/processors'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), './llm_factory/utils'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), './llm_factory'))
    
    # Direct imports to avoid relative import issues
    import dry_run_logger
    import cot_processor
    import standard_processor
    
    ChainOfThoughtProcessor = cot_processor.ChainOfThoughtProcessor
    StandardPromptProcessor = standard_processor.StandardPromptProcessor

# Configure logging to see our dry-run outputs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MockClient:
    """Mock client for testing without actual API calls"""
    def __init__(self):
        self.model_name = "mock-model"
        
    def generate_completion(self, **kwargs):
        return {"mock": "response", "config_used": kwargs}


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
        print(f"Cleaned up {len(files)} existing dry-run files")


def count_dry_run_files():
    """Count the number of JSON files in the dry-run folder"""
    dry_run_folder = "llm_factory_dry_run_responses"
    if not os.path.exists(dry_run_folder):
        return 0
    
    files = glob.glob(os.path.join(dry_run_folder, "*.json"))
    return len(files)


def inspect_dry_run_files():
    """Inspect the contents of dry-run files"""
    dry_run_folder = "llm_factory_dry_run_responses"
    if not os.path.exists(dry_run_folder):
        return []
    
    files = glob.glob(os.path.join(dry_run_folder, "*.json"))
    file_info = []
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_info.append({
                "filename": os.path.basename(file_path),
                "pipeline_name": data.get("pipeline_info", {}).get("pipeline_name"),
                "step_name": data.get("pipeline_info", {}).get("step_name"),
                "step_type": data.get("pipeline_info", {}).get("step_type"),
                "has_request_data": "request_data" in data,
                "has_metadata": "metadata" in data,
                "timestamp": data.get("session_info", {}).get("timestamp")
            })
        except Exception as e:
            file_info.append({
                "filename": os.path.basename(file_path),
                "error": str(e)
            })
    
    return sorted(file_info, key=lambda x: x.get("timestamp", ""))


def test_cot_dry_run_logging():
    """Test CoT processor dry-run file logging"""
    print("\n" + "="*60)
    print("Testing CoT Processor Dry-Run File Logging")
    print("="*60)
    
    # Clean up before test
    cleanup_dry_run_folder()
    
    # Test configuration with dry-run enabled
    cot_config = {
        "name": "file_logging_test_cot",
        "model": "gpt-4-test",
        "temperature": 0.8,
        "max_tokens": 500,
        "dry_run": True,
        "context_data": "Test context data for file logging",
        "steps": [
            {
                "type": "initialPrompt",
                "name": "test_step_1",
                "prompt": "Analyze the context data.",
                "output_key": "analysis"
            },
            {
                "type": "newProblem",
                "name": "test_step_2", 
                "prompt": "Generate a response based on analysis.",
                "input_key": "analysis",
                "output_key": "response",
                "temperature": 0.3
            },
            {
                "type": "finalAnswer",
                "name": "test_step_3",
                "prompt": "Provide final conclusion.",
                "input_key": "response",
                "output_key": "conclusion"
            }
        ]
    }
    
    client = MockClient()
    processor = ChainOfThoughtProcessor()
    
    try:
        files_before = count_dry_run_files()
        result = processor.process(client, cot_config)
        files_after = count_dry_run_files()
        
        files_created = files_after - files_before
        
        print(f"✅ CoT processor executed successfully")
        print(f"📁 Files created: {files_created}")
        print(f"📊 Total files in dry-run folder: {files_after}")
        
        # Inspect the created files
        file_info = inspect_dry_run_files()
        
        # We expect: 3 step files + 1 summary file = 4 files
        expected_files = 4
        if files_created >= expected_files:
            print(f"✅ Expected file count met ({files_created} >= {expected_files})")
            
            # Check for summary file
            summary_files = [f for f in file_info if "SUMMARY" in f.get("filename", "")]
            if summary_files:
                print(f"✅ Pipeline summary file created: {summary_files[0]['filename']}")
            else:
                print(f"⚠️  No summary file found")
            
            # Check step files
            step_files = [f for f in file_info if "SUMMARY" not in f.get("filename", "")]
            print(f"✅ Step files created: {len(step_files)}")
            
            return True
        else:
            print(f"❌ Expected at least {expected_files} files, got {files_created}")
            return False
        
    except Exception as e:
        print(f"❌ CoT dry-run logging test failed: {e}")
        return False


def test_standard_dry_run_logging():
    """Test Standard processor dry-run file logging"""
    print("\n" + "="*60)
    print("Testing Standard Processor Dry-Run File Logging")
    print("="*60)
    
    # Test configuration with dry-run enabled
    standard_config = {
        "name": "file_logging_test_standard",
        "model": "claude-3-test",
        "temperature": 0.2,
        "max_tokens": 300,
        "dry_run": True,
        "prompt": "Extract key information from the context.",
        "context_data": "Sample document content for file logging test",
        "schema": {
            "type": "object",
            "properties": {
                "key_points": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
    
    client = MockClient()
    processor = StandardPromptProcessor()
    
    try:
        files_before = count_dry_run_files()
        result = processor.process(client, standard_config)
        files_after = count_dry_run_files()
        
        files_created = files_after - files_before
        
        print(f"✅ Standard processor executed successfully")
        print(f"📁 Files created: {files_created}")
        print(f"📊 Total files in dry-run folder: {files_after}")
        
        # For standard processor, we expect: 1 extraction file + 1 summary file = 2 files
        expected_files = 2
        if files_created >= expected_files:
            print(f"✅ Expected file count met ({files_created} >= {expected_files})")
            return True
        else:
            print(f"❌ Expected at least {expected_files} files, got {files_created}")
            return False
        
    except Exception as e:
        print(f"❌ Standard dry-run logging test failed: {e}")
        return False


def test_file_contents():
    """Test that dry-run files contain expected data structure"""
    print("\n" + "="*60)
    print("Testing Dry-Run File Contents")
    print("="*60)
    
    file_info = inspect_dry_run_files()
    
    if not file_info:
        print("❌ No files found to inspect")
        return False
    
    print(f"📋 Inspecting {len(file_info)} files:")
    
    all_valid = True
    for info in file_info:
        filename = info.get("filename", "unknown")
        
        if "error" in info:
            print(f"❌ {filename}: {info['error']}")
            all_valid = False
            continue
        
        # Check required fields
        required_checks = [
            ("pipeline_name", info.get("pipeline_name")),
            ("step_name", info.get("step_name")),
            ("step_type", info.get("step_type")),
            ("has_request_data", info.get("has_request_data")),
            ("has_metadata", info.get("has_metadata")),
            ("timestamp", info.get("timestamp"))
        ]
        
        file_valid = True
        for field_name, value in required_checks:
            if not value:
                print(f"⚠️  {filename}: Missing {field_name}")
                file_valid = False
        
        if file_valid:
            print(f"✅ {filename}: Valid structure")
        else:
            all_valid = False
    
    return all_valid


def main():
    """Run all dry-run file logging tests"""
    print("🚀 Starting Dry-Run File Logging Test Suite")
    print("This test verifies that dry-run mode creates proper JSON log files.\n")
    
    tests = [
        ("CoT Processor File Logging", test_cot_dry_run_logging),
        ("Standard Processor File Logging", test_standard_dry_run_logging),
        ("File Contents Validation", test_file_contents)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
    
    # Print final file listing
    print("\n" + "="*60)
    print("FINAL DRY-RUN FILES CREATED")
    print("="*60)
    
    file_info = inspect_dry_run_files()
    if file_info:
        for info in file_info:
            filename = info.get("filename", "unknown")
            pipeline = info.get("pipeline_name", "unknown")
            step = info.get("step_name", "unknown")
            step_type = info.get("step_type", "unknown")
            print(f"📄 {filename}")
            print(f"   Pipeline: {pipeline} | Step: {step} | Type: {step_type}")
    else:
        print("No files found in dry-run folder")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<40} {status}")
        if not success:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED! Dry-run file logging is working correctly.")
        print("✅ JSON files are being created with proper timestamps.")
        print("✅ File structure and content validation successful.")
        
        dry_run_folder = "llm_factory_dry_run_responses"
        if os.path.exists(dry_run_folder):
            print(f"📁 Check the files in: {os.path.abspath(dry_run_folder)}")
        
        return 0
    else:
        print("💥 SOME TESTS FAILED! Please review the dry-run logging implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())