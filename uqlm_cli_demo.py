#!/usr/bin/env python3
"""
UQLM Command Line Demo

A simple command-line interface to test UQLM uncertainty quantification methods
with Google's Gemini models.
"""

import asyncio
import os
import sys
import json
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime

# Import UQLM components
try:
    from uqlm import BlackBoxUQ, WhiteBoxUQ, LLMPanel, UQEnsemble, SemanticEntropy
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, HumanMessage
    from langchain_core.outputs import ChatResult, ChatGeneration, LLMResult
    from gemini_whitebox_llm import create_gemini_whitebox_llm
    import google.generativeai as genai
    UQLM_AVAILABLE = True
    print("✅ UQLM library imported successfully")
except ImportError as e:
    print(f"❌ UQLM import error: {e}")
    UQLM_AVAILABLE = False

class UQLMCLIDemo:
    """Command-line demo for UQLM uncertainty quantification methods."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.llm = None
        self.model_name = model_name
        self.setup_llm()
    
    def setup_llm(self):
        """Set up the Gemini LLM through Google Generative AI API."""
        # Check for API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("⚠️ GOOGLE_API_KEY environment variable not set. Please set your Google API key.")
            
        try:
            print(f"🔧 Setting up Gemini model: {self.model_name}")
            
            # Create the UQLMCompatibleLLM wrapper
            self.llm = self.UQLMCompatibleLLM(
                model=self.model_name,
                temperature=0.7,
                google_api_key=api_key
            )
            print("✅ Gemini model initialized successfully!")
        except Exception as e:
            print(f"❌ Error setting up Gemini model: {e}")
            raise
    

    
    async def run_blackbox_uq(self, prompts: List[str], num_responses: int = 5) -> Dict[str, Any]:
        """Run BlackBox Uncertainty Quantification."""
        print(f"🔄 Running BlackBox UQ with {num_responses} responses...")
        
        try:
            bbuq = BlackBoxUQ(
                llm=self.llm, 
                scorers=["semantic_negentropy", "exact_match", "noncontradiction", "bert_score"],
                use_best=True
            )
            results = await bbuq.generate_and_score(prompts=prompts, num_responses=num_responses)
            return {
                "results": results,
                "method": "blackbox",
                "success": True
            }
        except Exception as e:
            print(f"❌ BlackBox UQ error: {e}")
            return {"error": str(e), "method": "blackbox", "success": False}
    
    async def run_whitebox_uq(self, prompts: List[str]) -> Dict[str, Any]:
        """Run WhiteBox Uncertainty Quantification with real log probabilities."""
        print("🔄 Running WhiteBox UQ with real log probabilities...")
        
        try:
            wbuq = WhiteBoxUQ(
                llm=self.llm,
                scorers=["min_probability"]
            )
            results = await wbuq.generate_and_score(prompts=prompts)
            return {
                "results": results,
                "method": "whitebox",
                "success": True
            }
        except Exception as e:
            print(f"❌ WhiteBox UQ error: {e}")
            return {"error": str(e), "method": "whitebox", "success": False}
    
    async def run_llm_panel(self, prompts: List[str]) -> Dict[str, Any]:
        """Run LLM Panel (LLM-as-a-Judge) methods."""
        print("🔄 Running LLM Panel...")
        
        try:
            # Create multiple judge instances with different Gemini models
            api_key = os.getenv('GOOGLE_API_KEY')
            judge1 = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.3, google_api_key=api_key)
            judge2 = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.5, google_api_key=api_key)
            judge3 = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.7, google_api_key=api_key)
            
            panel = LLMPanel(llm=self.llm, judges=[judge1, judge2, judge3])
            results = await panel.generate_and_score(prompts=prompts)
            return {
                "results": results,
                "method": "judges",
                "success": True
            }
        except Exception as e:
            print(f"❌ LLM Panel error: {e}")
            return {"error": str(e), "method": "judges", "success": False}
    
    async def run_ensemble(self, prompts: List[str]) -> Dict[str, Any]:
        """Run UQ Ensemble methods."""
        print("🔄 Running UQ Ensemble...")
        
        try:
            # Off-the-shelf ensemble
            uqe = UQEnsemble(llm=self.llm)
            results = await uqe.generate_and_score(prompts=prompts, num_responses=5)
            return {
                "results": results,
                "method": "ensemble",
                "success": True
            }
        except Exception as e:
            print(f"❌ Ensemble UQ error: {e}")
            return {"error": str(e), "method": "ensemble", "success": False}
    
    async def run_semantic_entropy(self, prompts: List[str]) -> Dict[str, Any]:
        """Run Semantic Entropy methods."""
        print("🔄 Running Semantic Entropy...")
        
        try:
            se = SemanticEntropy(llm=self.llm)
            results = await se.generate_and_score(prompts=prompts)
            return {
                "results": results,
                "method": "semantic_entropy",
                "success": True
            }
        except Exception as e:
            print(f"❌ Semantic Entropy error: {e}")
            return {"error": str(e), "method": "semantic_entropy", "success": False}
    
    def display_results(self, results: Dict[str, Any], method_name: str):
        """Display results in a formatted way."""
        print(f"\n{'='*60}")
        print(f"📊 {method_name.upper()} RESULTS")
        print(f"{'='*60}")
        
        if results.get("success", False) and "results" in results:
            # Real UQLM results
            uqlm_results = results["results"]
            print(f"✅ Method: {results['method']}")
            
            if hasattr(uqlm_results, 'to_df'):
                df = uqlm_results.to_df()
                print(f"📊 Results DataFrame:")
                print(df.to_string())
            else:
                print(f"📊 Results: {uqlm_results}")
                
        else:
            print(f"❌ Error in {method_name}: {results.get('error', 'Unknown error')}")
    
    async def run_method(self, method: str, prompts: List[str], num_responses: int = 5):
        """Run a specific UQ method."""
        method_map = {
            "blackbox": self.run_blackbox_uq,
            "whitebox": self.run_whitebox_uq,
            "judges": self.run_llm_panel,
            "ensemble": self.run_ensemble,
            "semantic": self.run_semantic_entropy
        }
        
        if method not in method_map:
            print(f"❌ Unknown method: {method}")
            return None
        
        if method == "blackbox":
            return await method_map[method](prompts, num_responses)
        else:
            return await method_map[method](prompts)

def get_example_prompts() -> Dict[str, List[str]]:
    """Get example prompts for testing."""
    return {
        "science": [
            "What are the main causes of climate change?",
            "Explain quantum computing in simple terms.",
            "How does photosynthesis work in plants?"
        ],
        "history": [
            "What caused World War I?",
            "Who was the first person to walk on the moon?",
            "When did the Berlin Wall fall?"
        ],
        "technology": [
            "What is artificial intelligence?",
            "How do neural networks work?",
            "What is blockchain technology?"
        ],
        "general": [
            "What is the capital of France?",
            "How many continents are there?",
            "What is the largest ocean on Earth?"
        ]
    }

def main():
    """Main CLI application."""
    parser = argparse.ArgumentParser(description="UQLM CLI Demo - Uncertainty Quantification for Language Models")
    parser.add_argument("--method", "-m", 
                       choices=["blackbox", "whitebox", "judges", "ensemble", "semantic", "all"],
                       default="blackbox",
                       help="UQ method to run")
    parser.add_argument("--prompt", "-p", type=str,
                       help="Custom prompt to analyze")
    parser.add_argument("--examples", "-e", 
                       choices=["science", "history", "technology", "general"],
                       help="Use example prompts from category")
    parser.add_argument("--num-responses", "-n", type=int, default=5,
                       help="Number of responses for BlackBox UQ (default: 5)")
    parser.add_argument("--model", "-M", type=str, default="gemini-2.0-flash",
                       help="Gemini model name (default: gemini-2.0-flash)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode")
    
    args = parser.parse_args()
    
    if not UQLM_AVAILABLE:
        print("❌ UQLM library is not available. Please install it first.")
        return
    
    print("🎯 UQLM CLI Demo - Uncertainty Quantification for Language Models")
    print("="*70)
    
    # Initialize demo
    demo = UQLMCLIDemo(model_name=args.model)
    
    # Determine prompts to use
    prompts = []
    
    if args.interactive:
        print("\n🔧 Interactive Mode")
        print("Enter prompts (one per line, empty line to finish):")
        while True:
            prompt = input("> ").strip()
            if not prompt:
                break
            prompts.append(prompt)
    elif args.prompt:
        prompts = [args.prompt]
    elif args.examples:
        example_prompts = get_example_prompts()
        prompts = example_prompts[args.examples]
        print(f"\n📝 Using {args.examples} example prompts:")
        for i, prompt in enumerate(prompts, 1):
            print(f"   {i}. {prompt}")
    else:
        # Default prompt
        prompts = ["What are the main causes of climate change?"]
        print(f"\n📝 Using default prompt: {prompts[0]}")
    
    if not prompts:
        print("❌ No prompts provided. Use --prompt, --examples, or --interactive.")
        return
    
    print(f"\n🚀 Running analysis on {len(prompts)} prompt(s)...")
    
    # Run selected method(s)
    async def run_analysis():
        if args.method == "all":
            methods = ["blackbox", "whitebox", "judges", "ensemble", "semantic"]
        else:
            methods = [args.method]
        
        for method in methods:
            try:
                results = await demo.run_method(method, prompts, args.num_responses)
                if results:
                    demo.display_results(results, method)
            except Exception as e:
                print(f"❌ Error running {method}: {e}")
    
    # Run the analysis
    asyncio.run(run_analysis())
    
    print(f"\n{'='*70}")
    print("✅ Analysis complete!")
    print("\nFor more information about UQLM, visit:")
    print("🔗 https://github.com/cvs-health/uqlm")

if __name__ == "__main__":
    main()