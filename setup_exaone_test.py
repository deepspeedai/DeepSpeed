#!/usr/bin/env python3
"""
Setup script for EXAONE 4.0 testing environment

This script helps set up the environment for testing EXAONE 4.0 with DeepSpeed inference v2.
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ is required")
        return False
    
    logger.info("✓ Python version is compatible")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "torch",
        "transformers",
        "deepspeed"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"⚠ {package} is not installed")
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages with:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_transformers_version():
    """Check if transformers version supports EXAONE 4.0"""
    try:
        import transformers
        version = transformers.__version__
        logger.info(f"Transformers version: {version}")
        
        # EXAONE 4.0 requires transformers >= 4.54.0
        from packaging import version as pkg_version
        if pkg_version.parse(version) < pkg_version.parse("4.54.0"):
            logger.error("❌ Transformers 4.54.0+ is required for EXAONE 4.0")
            logger.info("Please upgrade transformers with:")
            logger.info("pip install --upgrade transformers")
            return False
        
        logger.info("✓ Transformers version is compatible")
        return True
        
    except ImportError:
        logger.error("❌ Could not check transformers version")
        return False

def check_deepspeed_installation():
    """Check if DeepSpeed is properly installed"""
    try:
        import deepspeed
        logger.info(f"DeepSpeed version: {deepspeed.__version__}")
        
        # Check if we're in the DeepSpeed directory
        if os.path.exists("deepspeed"):
            logger.info("✓ DeepSpeed source directory found")
            return True
        else:
            logger.warning("⚠ DeepSpeed source directory not found")
            logger.info("Make sure you're running this from the DeepSpeed root directory")
            return False
            
    except ImportError:
        logger.error("❌ DeepSpeed is not installed")
        return False

def test_exaone_config_loading():
    """Test if EXAONE 4.0 config can be loaded"""
    try:
        from transformers import AutoConfig
        
        # Test 1.2B model config
        config_1_2b = AutoConfig.from_pretrained("LGAI-EXAONE/EXAONE-4.0-1.2B", trust_remote_code=True)
        logger.info("✓ EXAONE 4.0 1.2B config loaded successfully")
        
        # Test 32B model config
        config_32b = AutoConfig.from_pretrained("LGAI-EXAONE/EXAONE-4.0-32B", trust_remote_code=True)
        logger.info("✓ EXAONE 4.0 32B config loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load EXAONE 4.0 config: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Setting up EXAONE 4.0 testing environment...")
    logger.info("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Transformers Version", check_transformers_version),
        ("DeepSpeed Installation", check_deepspeed_installation),
        ("EXAONE Config Loading", test_exaone_config_loading),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        if not check_func():
            all_passed = False
    
    logger.info("\n" + "=" * 50)
    if all_passed:
        logger.info("🎉 Environment setup completed successfully!")
        logger.info("You can now run the test script with:")
        logger.info("python test_exaone_inference.py")
    else:
        logger.error("💥 Environment setup failed!")
        logger.info("Please fix the issues above before running the test script.")

if __name__ == "__main__":
    main()
