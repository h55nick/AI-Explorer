#!/usr/bin/env python3
"""
UQLM Demo Examples

This script demonstrates various usage patterns and examples for the UQLM demo applications.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uqlm_cli_demo import UQLMCLIDemo

class UQLMExamples:
    """Collection of UQLM usage examples."""
    
    def __init__(self):
        self.demo = UQLMCLIDemo(demo_mode=True)
    
    async def example_1_basic_usage(self):
        """Example 1: Basic uncertainty quantification."""
        print("📝 Example 1: Basic Uncertainty Quantification")
        print("-" * 50)
        
        prompt = "What is the capital of France?"
        print(f"Prompt: {prompt}")
        
        # Run BlackBox UQ
        results = await self.demo.run_method("blackbox", [prompt])
        if results:
            print(f"BlackBox UQ Confidence: {results['confidence']:.3f}")
            print(f"Response: {results['response'][:100]}...")
        
        print()
    
    async def example_2_method_comparison(self):
        """Example 2: Compare different methods on the same prompt."""
        print("📊 Example 2: Method Comparison")
        print("-" * 50)
        
        prompt = "Explain the theory of relativity in simple terms."
        print(f"Prompt: {prompt}")
        print()
        
        methods = ["blackbox", "whitebox", "judges", "ensemble", "semantic"]
        results = {}
        
        for method in methods:
            result = await self.demo.run_method(method, [prompt])
            if result:
                results[method] = result['confidence']
                print(f"{method.capitalize():12}: {result['confidence']:.3f}")
        
        if results:
            best_method = max(results, key=results.get)
            worst_method = min(results, key=results.get)
            print(f"\nHighest confidence: {best_method} ({results[best_method]:.3f})")
            print(f"Lowest confidence:  {worst_method} ({results[worst_method]:.3f})")
        
        print()
    
    async def example_3_factual_vs_creative(self):
        """Example 3: Compare factual vs creative prompts."""
        print("🎭 Example 3: Factual vs Creative Prompts")
        print("-" * 50)
        
        prompts = {
            "Factual": "What is 2 + 2?",
            "Creative": "Write a short poem about the ocean."
        }
        
        for prompt_type, prompt in prompts.items():
            print(f"{prompt_type} prompt: {prompt}")
            
            # Test with BlackBox UQ
            result = await self.demo.run_method("blackbox", [prompt])
            if result:
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Response: {result['response'][:80]}...")
            print()
    
    async def example_4_uncertainty_patterns(self):
        """Example 4: Analyze uncertainty patterns across different domains."""
        print("🔬 Example 4: Uncertainty Patterns by Domain")
        print("-" * 50)
        
        domains = {
            "Mathematics": "What is the derivative of x^2?",
            "History": "When did World War II end?",
            "Science": "What causes photosynthesis?",
            "Philosophy": "What is the meaning of life?",
            "Current Events": "What happened in the news yesterday?",
            "Speculation": "What will technology look like in 2050?"
        }
        
        domain_confidences = {}
        
        for domain, prompt in domains.items():
            result = await self.demo.run_method("ensemble", [prompt])
            if result:
                confidence = result['confidence']
                domain_confidences[domain] = confidence
                print(f"{domain:15}: {confidence:.3f} - {prompt[:40]}...")
        
        if domain_confidences:
            print("\nDomain Confidence Ranking:")
            sorted_domains = sorted(domain_confidences.items(), key=lambda x: x[1], reverse=True)
            for i, (domain, confidence) in enumerate(sorted_domains, 1):
                print(f"  {i}. {domain}: {confidence:.3f}")
        
        print()
    
    async def example_5_confidence_thresholds(self):
        """Example 5: Demonstrate confidence threshold usage."""
        print("🎯 Example 5: Confidence Thresholds")
        print("-" * 50)
        
        prompts = [
            "What is the speed of light?",
            "How do you make a perfect pizza?",
            "What will the weather be like tomorrow?",
            "Explain quantum entanglement.",
            "What is 1 + 1?"
        ]
        
        thresholds = {
            "High Confidence": 0.8,
            "Medium Confidence": 0.6,
            "Low Confidence": 0.4
        }
        
        results = []
        for prompt in prompts:
            result = await self.demo.run_method("blackbox", [prompt])
            if result:
                results.append((prompt, result['confidence']))
        
        print("Confidence Analysis:")
        for threshold_name, threshold_value in thresholds.items():
            print(f"\n{threshold_name} (≥{threshold_value}):")
            matching = [r for r in results if r[1] >= threshold_value]
            if matching:
                for prompt, confidence in matching:
                    print(f"  • {confidence:.3f}: {prompt}")
            else:
                print("  • No prompts in this category")
        
        print()
    
    async def example_6_batch_processing(self):
        """Example 6: Batch processing multiple prompts."""
        print("📦 Example 6: Batch Processing")
        print("-" * 50)
        
        batch_prompts = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What is deep learning?",
            "Explain neural networks.",
            "What is natural language processing?"
        ]
        
        print(f"Processing {len(batch_prompts)} prompts...")
        
        batch_results = []
        for i, prompt in enumerate(batch_prompts, 1):
            print(f"  {i}/{len(batch_prompts)}: Processing...")
            result = await self.demo.run_method("whitebox", [prompt])
            if result:
                batch_results.append({
                    'prompt': prompt,
                    'confidence': result['confidence'],
                    'response': result['response']
                })
        
        print(f"\nBatch Results Summary:")
        if batch_results:
            avg_confidence = sum(r['confidence'] for r in batch_results) / len(batch_results)
            max_confidence = max(r['confidence'] for r in batch_results)
            min_confidence = min(r['confidence'] for r in batch_results)
            
            print(f"  • Total processed: {len(batch_results)}")
            print(f"  • Average confidence: {avg_confidence:.3f}")
            print(f"  • Max confidence: {max_confidence:.3f}")
            print(f"  • Min confidence: {min_confidence:.3f}")
            
            print(f"\nTop 3 Most Confident:")
            sorted_results = sorted(batch_results, key=lambda x: x['confidence'], reverse=True)
            for i, result in enumerate(sorted_results[:3], 1):
                print(f"  {i}. {result['confidence']:.3f}: {result['prompt']}")
        
        print()

async def run_all_examples():
    """Run all examples."""
    print("🎯 UQLM Demo Examples")
    print("=" * 70)
    print("This script demonstrates various usage patterns for UQLM uncertainty quantification.")
    print("All examples run in demo mode with mock responses.\n")
    
    examples = UQLMExamples()
    
    # Run all examples
    await examples.example_1_basic_usage()
    await examples.example_2_method_comparison()
    await examples.example_3_factual_vs_creative()
    await examples.example_4_uncertainty_patterns()
    await examples.example_5_confidence_thresholds()
    await examples.example_6_batch_processing()
    
    print("=" * 70)
    print("✅ All examples completed!")
    print("\n🚀 Try these commands:")
    print("   • python uqlm_cli_demo.py --help")
    print("   • python uqlm_cli_demo.py --demo-mode --method all")
    print("   • streamlit run uqlm_demo.py")

def main():
    """Main function."""
    asyncio.run(run_all_examples())

if __name__ == "__main__":
    main()