#!/usr/bin/env python3
"""
Test script to verify the GRPO training setup and dependencies
"""

import sys
import importlib
import torch


def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'trl',
        'bitsandbytes',
        'peft',
        'trackio',
        'numpy',
        'scipy',
        'sklearn',
    ]
    
    print("🔍 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError as e:
            print(f"   ❌ {package}: {str(e)}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages imported successfully!")
        return True


def test_gpu():
    """Test GPU availability and configuration"""
    print("\n🔍 Testing GPU environment...")
    
    if not torch.cuda.is_available():
        print("   ⚠️  CUDA not available. Training will be slow on CPU.")
        print("   Consider using a GPU-enabled environment for optimal performance.")
        return False
    
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    print(f"   ✅ Number of GPUs: {torch.cuda.device_count()}")
    print(f"   ✅ Current GPU: {torch.cuda.current_device()}")
    print(f"   ✅ GPU name: {torch.cuda.get_device_name()}")
    print(f"   ✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True


def test_model_loading():
    """Test if we can load a small model for verification"""
    print("\n🔍 Testing model loading...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test with a small model
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        print("   ✅ Successfully loaded test tokenizer")
        
        # Test tokenization
        test_text = "This is a test."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   ✅ Tokenization works: {len(tokens['input_ids'][0])} tokens")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {str(e)}")
        return False


def test_dataset_loading():
    """Test if we can load the GSM8K dataset"""
    print("\n🔍 Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Load a small subset of GSM8K
        dataset = load_dataset("openai/gsm8k", "main", split="train[:5]")
        print(f"   ✅ Successfully loaded GSM8K dataset: {len(dataset)} examples")
        
        # Check dataset structure
        example = dataset[0]
        print(f"   ✅ Dataset structure: {list(example.keys())}")
        print(f"   ✅ Sample question: {example['question'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {str(e)}")
        return False


def test_grpo_components():
    """Test if our GRPO components can be imported and initialized"""
    print("\n🔍 Testing GRPO components...")
    
    try:
        from grpo_training import (
            GRPOTrainingConfig,
            RewardFunctions,
            DatasetProcessor,
            ModelSetup
        )
        
        # Test configuration
        config = GRPOTrainingConfig()
        print("   ✅ GRPOTrainingConfig initialized")
        
        # Test reward functions
        reward_functions = RewardFunctions()
        print("   ✅ RewardFunctions initialized")
        
        # Test dataset processor
        dataset_processor = DatasetProcessor(reward_functions)
        print("   ✅ DatasetProcessor initialized")
        
        # Test model setup
        model_setup = ModelSetup(config)
        print("   ✅ ModelSetup initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GRPO components test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("🚀 GRPO Training Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("GPU Environment", test_gpu),
        ("Model Loading", test_model_loading),
        ("Dataset Loading", test_dataset_loading),
        ("GRPO Components", test_grpo_components),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready for GRPO training.")
        print("\nTo start training, run:")
        print("   python run_training.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()