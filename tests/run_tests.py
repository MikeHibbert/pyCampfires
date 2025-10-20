"""
Test runner for multimodal functionality tests.
"""

import sys
import os
import pytest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Run all multimodal tests."""
    test_dir = Path(__file__).parent
    
    # Test files to run
    test_files = [
        "test_multimodal_torch.py",
        "test_multimodal_openrouter.py",
        "test_audio_processing.py",
        "test_metadata_extractor.py",
        "test_multimodal_local_driver.py",
        "test_multimodal_prompts.py"
    ]
    
    print("Running Campfires Multimodal Tests")
    print("=" * 50)
    
    # Run each test file
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            print(f"\nRunning {test_file}...")
            result = pytest.main([str(test_path), "-v"])
            if result != 0:
                print(f"❌ Tests failed in {test_file}")
                return result
            else:
                print(f"✅ All tests passed in {test_file}")
        else:
            print(f"⚠️  Test file {test_file} not found")
    
    print("\n" + "=" * 50)
    print("🎉 All multimodal tests completed successfully!")
    return 0


def run_specific_test(test_name):
    """Run a specific test file."""
    test_dir = Path(__file__).parent
    test_path = test_dir / f"test_{test_name}.py"
    
    if not test_path.exists():
        print(f"❌ Test file test_{test_name}.py not found")
        return 1
    
    print(f"Running test_{test_name}.py...")
    result = pytest.main([str(test_path), "-v"])
    
    if result == 0:
        print(f"✅ All tests passed in test_{test_name}.py")
    else:
        print(f"❌ Tests failed in test_{test_name}.py")
    
    return result


def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    test_dir = Path(__file__).parent
    
    print("Running tests with coverage...")
    result = pytest.main([
        str(test_dir),
        "--cov=campfires",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ])
    
    if result == 0:
        print("✅ All tests passed with coverage report generated")
    else:
        print("❌ Some tests failed")
    
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "coverage":
            exit_code = run_tests_with_coverage()
        elif command.startswith("test_"):
            # Remove "test_" prefix if provided
            test_name = command[5:] if command.startswith("test_") else command
            exit_code = run_specific_test(test_name)
        else:
            # Treat as test name
            exit_code = run_specific_test(command)
    else:
        exit_code = run_all_tests()
    
    sys.exit(exit_code)