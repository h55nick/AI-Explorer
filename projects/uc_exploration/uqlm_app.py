#!/usr/bin/env python3
"""
UQLM (Uncertainty Quantification for Language Models) Application

This application provides access to all the different uncertainty quantification methods
available in UQLM using Google's Gemini models.

Features:
- BlackBox UQ (consistency-based methods)
- WhiteBox UQ (token probability-based methods)
- LLM-as-a-Judge methods
- Ensemble methods
- Semantic Entropy methods
- Interactive Streamlit interface
"""

import streamlit as st
import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Optional seaborn import
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Import UQLM components
try:
    from uqlm import BlackBoxUQ, WhiteBoxUQ, LLMPanel, UQEnsemble, SemanticEntropy
    from langchain_google_genai import ChatGoogleGenerativeAI
    from gemini_whitebox_llm import create_gemini_whitebox_llm
    import google.generativeai as genai
    UQLM_AVAILABLE = True
except ImportError as e:
    st.error(f"UQLM import error: {e}")
    UQLM_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="UQLM - Uncertainty Quantification for Language Models",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .method-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .confidence-score {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class UQLMApp:
    """Main application class for UQLM uncertainty quantification methods."""
    
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.llm = None
        self.model_name = model_name
        self.setup_llm()
    
    def setup_llm(self):
        """Set up the Gemini LLM through Google Generative AI API."""
        # Check for API key
        api_key = st.session_state.get('google_api_key') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("❌ Google API Key is required. Please provide your API key.")
            st.stop()
            
        try:
            # Configure the API key
            genai.configure(api_key=api_key)
            
            # Initialize ChatGoogleGenerativeAI with the specified model
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.7,
                max_tokens=1024,
                google_api_key=api_key
            )
            st.success(f"✅ {self.model_name} model initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize Gemini API: {e}")
            st.stop()
    

    async def run_blackbox_uq(self, prompts: List[str], num_responses: int = 5) -> Dict[str, Any]:
        """Run BlackBox Uncertainty Quantification."""
        
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
            st.error(f"BlackBox UQ error: {e}")
            return {"error": str(e), "method": "blackbox", "success": False}
    
    async def run_whitebox_uq(self, prompts: List[str]) -> Dict[str, Any]:
        """Run WhiteBox Uncertainty Quantification with real log probabilities."""
        
        try:
            # Get API key
            api_key = st.session_state.get('google_api_key') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                st.error("❌ API key is required for WhiteBox UQ")
                return {"error": "API key required", "method": "whitebox", "success": False}
            
            # Create the custom LLM with logprobs support
            gemini_llm = create_gemini_whitebox_llm(
                api_key=api_key,
                model_name=self.model_name,
                temperature=0.7,
                max_tokens=1024
            )
            
            # Create a wrapper that UQLM can use
            class UQLMCompatibleLLM:
                def __init__(self, gemini_llm):
                    self.gemini_llm = gemini_llm
                    self.logprobs = True  # UQLM checks for this attribute
                    # Add all attributes that UQLM might expect
                    self.temperature = 0.7
                    self.max_tokens = 1024
                    self.model_name = 'gemini-1.5-flash'
                
                def generate(self, prompts, **kwargs):
                    """Generate responses for UQLM."""
                    results = []
                    for prompt in prompts:
                        result = self.gemini_llm.generate(prompt, **kwargs)
                        # Format for UQLM compatibility
                        generation_result = type('GenerationResult', (), {
                            'generations': [[type('Generation', (), {
                                'text': result['text'],
                                'generation_info': {'logprobs': result['logprobs']}
                            })()]]
                        })()
                        results.append(generation_result)
                    return results
                
                def __call__(self, prompt, **kwargs):
                    """Direct call interface."""
                    result = self.gemini_llm.generate(prompt, **kwargs)
                    return result['text']
            
            compatible_llm = UQLMCompatibleLLM(gemini_llm)
            
            wbuq = WhiteBoxUQ(
                llm=compatible_llm,
                scorers=["min_probability", "length_normalized"]
            )
            results = await wbuq.generate_and_score(prompts=prompts)
            return {
                "results": results,
                "method": "whitebox",
                "success": True
            }
        except Exception as e:
            st.error(f"WhiteBox UQ error: {e}")
            return {"error": str(e), "method": "whitebox", "success": False}
    
    async def run_llm_panel(self, prompts: List[str]) -> Dict[str, Any]:
        """Run LLM Panel (LLM-as-a-Judge) methods."""
        
        try:
            # Create multiple judge instances
            api_key = st.session_state.get('google_api_key') or os.getenv('GOOGLE_API_KEY')
            judge1 = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.3, google_api_key=api_key)
            judge2 = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.5, google_api_key=api_key)
            judge3 = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.7, google_api_key=api_key)
            
            panel = LLMPanel(llm=self.llm, judges=[judge1, judge2, judge3])
            results = await panel.generate_and_score(prompts=prompts)
            return {
                "results": results,
                "method": "judges",
                "success": True
            }
        except Exception as e:
            st.error(f"LLM Panel error: {e}")
            return {"error": str(e), "method": "judges", "success": False}
    
    async def run_ensemble(self, prompts: List[str]) -> Dict[str, Any]:
        """Run UQ Ensemble methods."""
        
        try:
            # Option 1: Off-the-shelf ensemble
            uqe = UQEnsemble(llm=self.llm)
            results = await uqe.generate_and_score(prompts=prompts, num_responses=5)
            return {
                "results": results,
                "method": "ensemble",
                "success": True
            }
        except Exception as e:
            st.error(f"Ensemble UQ error: {e}")
            return {"error": str(e), "method": "ensemble", "success": False}
    
    async def run_semantic_entropy(self, prompts: List[str]) -> Dict[str, Any]:
        """Run Semantic Entropy methods."""
        
        try:
            se = SemanticEntropy(llm=self.llm)
            results = await se.generate_and_score(prompts=prompts)
            return {
                "results": results,
                "method": "semantic_entropy",
                "success": True
            }
        except Exception as e:
            st.error(f"Semantic Entropy error: {e}")
            return {"error": str(e), "method": "semantic_entropy", "success": False}

def display_results(results: Dict[str, Any], method_name: str):
    """Display results in a formatted way."""
    st.subheader(f"📊 {method_name} Results")
    
    if results.get("success", False) and not isinstance(results.get("confidence"), type(None)):
        # Demo mode results
        if "confidence" in results:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="confidence-score">
                    Confidence: {results['confidence']:.3f}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write("**Response:**")
                st.write(results.get("response", "No response available"))
            
            # Display detailed scores
            if "scores" in results:
                st.write("**Detailed Scores:**")
                scores_df = pd.DataFrame(list(results["scores"].items()), 
                                       columns=["Metric", "Score"])
                st.dataframe(scores_df, use_container_width=True)
                
                # Create a bar chart of scores
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(scores_df["Metric"], scores_df["Score"])
                ax.set_ylabel("Score")
                ax.set_title(f"{method_name} Detailed Scores")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
    
    elif results.get("success", False) and "results" in results:
        # Real UQLM results
        uqlm_results = results["results"]
        if hasattr(uqlm_results, 'to_df'):
            df = uqlm_results.to_df()
            st.dataframe(df, use_container_width=True)
        else:
            st.write(uqlm_results)
    
    else:
        st.error(f"Error in {method_name}: {results.get('error', 'Unknown error')}")

def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🎯 UQLM: Uncertainty Quantification for Language Models</h1>', 
                unsafe_allow_html=True)
    
    if not UQLM_AVAILABLE:
        st.error("UQLM library is not available. Please install it first.")
        return
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Key input
    st.sidebar.subheader("🔑 Google API Key")
    api_key_input = st.sidebar.text_input(
        "Enter your Google API Key:",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey",
        placeholder="Enter API key to use real Gemini API"
    )
    
    if api_key_input:
        st.session_state['google_api_key'] = api_key_input
        st.sidebar.success("✅ API Key provided")
    else:
        st.sidebar.error("❌ API Key is required")
        st.stop()
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    model_name = st.sidebar.selectbox(
        "Select Gemini Model:",
        options=[
            "gemini-2.0-flash",
            "gemini-1.5-flash", 
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ],
        index=0,
        help="Choose which Gemini model to use"
    )
    
    # Initialize app with selected model
    app = UQLMApp(model_name=model_name)
    
    # Method selection
    methods = {
        "BlackBox UQ": "Consistency-based methods (multiple generations)",
        "WhiteBox UQ": "Token probability-based methods",
        "LLM Panel": "LLM-as-a-Judge methods",
        "UQ Ensemble": "Ensemble of multiple methods",
        "Semantic Entropy": "Semantic entropy-based methods"
    }
    
    selected_methods = st.sidebar.multiselect(
        "Select UQ Methods to Run:",
        options=list(methods.keys()),
        default=["BlackBox UQ"],
        help="Choose which uncertainty quantification methods to apply"
    )
    
    # Parameters
    num_responses = st.sidebar.slider(
        "Number of Responses (for BlackBox UQ):",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of responses to generate for consistency-based methods"
    )
    
    # Main input area
    st.header("📝 Input")
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Single prompt", "Multiple prompts", "Example prompts"]
    )
    
    prompts = []
    
    if input_method == "Single prompt":
        prompt = st.text_area(
            "Enter your prompt:",
            value="What are the main causes of climate change?",
            height=100,
            help="Enter a single prompt to analyze"
        )
        if prompt.strip():
            prompts = [prompt.strip()]
    
    elif input_method == "Multiple prompts":
        prompt_text = st.text_area(
            "Enter multiple prompts (one per line):",
            value="What are the main causes of climate change?\nExplain quantum computing in simple terms.\nWhat is the capital of France?",
            height=150,
            help="Enter multiple prompts, one per line"
        )
        prompts = [p.strip() for p in prompt_text.split('\n') if p.strip()]
    
    else:  # Example prompts
        example_categories = {
            "Science": [
                "What are the main causes of climate change?",
                "Explain quantum computing in simple terms.",
                "How does photosynthesis work?"
            ],
            "History": [
                "What caused World War I?",
                "Who was the first person to walk on the moon?",
                "When did the Berlin Wall fall?"
            ],
            "Technology": [
                "What is artificial intelligence?",
                "How do neural networks work?",
                "What is blockchain technology?"
            ],
            "General Knowledge": [
                "What is the capital of France?",
                "How many continents are there?",
                "What is the largest ocean on Earth?"
            ]
        }
        
        selected_category = st.selectbox("Choose example category:", list(example_categories.keys()))
        selected_examples = st.multiselect(
            "Select example prompts:",
            example_categories[selected_category],
            default=example_categories[selected_category][:1]
        )
        prompts = selected_examples
    
    # Display selected prompts
    if prompts:
        st.write(f"**Selected prompts ({len(prompts)}):**")
        for i, prompt in enumerate(prompts, 1):
            st.write(f"{i}. {prompt}")
    
    # Run analysis
    if st.button("🚀 Run Uncertainty Quantification", type="primary"):
        if not prompts:
            st.error("Please enter at least one prompt.")
            return
        
        if not selected_methods:
            st.error("Please select at least one UQ method.")
            return
        
        st.header("📊 Results")
        
        # Create tabs for different methods
        if len(selected_methods) > 1:
            tabs = st.tabs(selected_methods)
        else:
            tabs = [st.container()]
        
        # Run each selected method
        for i, method in enumerate(selected_methods):
            with tabs[i] if len(selected_methods) > 1 else tabs[0]:
                with st.spinner(f"Running {method}..."):
                    try:
                        # Use a different approach for async calls in Streamlit
                        import asyncio
                        import threading
                        
                        def run_async_method(coro):
                            """Run async method in a new thread with its own event loop."""
                            def run_in_thread():
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    return loop.run_until_complete(coro)
                                finally:
                                    loop.close()
                            
                            # Run in a separate thread to avoid event loop conflicts
                            import concurrent.futures
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(run_in_thread)
                                return future.result()
                        
                        if method == "BlackBox UQ":
                            results = run_async_method(app.run_blackbox_uq(prompts, num_responses))
                        elif method == "WhiteBox UQ":
                            results = run_async_method(app.run_whitebox_uq(prompts))
                        elif method == "LLM Panel":
                            results = run_async_method(app.run_llm_panel(prompts))
                        elif method == "UQ Ensemble":
                            results = run_async_method(app.run_ensemble(prompts))
                        elif method == "Semantic Entropy":
                            results = run_async_method(app.run_semantic_entropy(prompts))
                        
                        display_results(results, method)
                        
                    except Exception as e:
                        st.error(f"Error running {method}: {e}")
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
    
    # Information section
    st.header("ℹ️ About UQLM Methods")
    
    method_info = {
        "BlackBox UQ": {
            "description": "Treats the LLM as a black box and evaluates consistency of multiple responses.",
            "pros": ["Universal compatibility", "Intuitive approach", "No access to internals needed"],
            "cons": ["Higher latency", "More expensive", "Requires multiple generations"],
            "scorers": ["Semantic Negentropy", "Exact Match", "Non-contradiction", "BERT Score", "Cosine Similarity"]
        },
        "WhiteBox UQ": {
            "description": "Leverages token probabilities to estimate uncertainty.",
            "pros": ["Fast execution", "No extra cost", "Direct probability access"],
            "cons": ["Limited compatibility", "Requires token probabilities", "Model-dependent"],
            "scorers": ["Minimum Token Probability", "Length-Normalized Joint Probability"]
        },
        "LLM Panel": {
            "description": "Uses multiple LLMs as judges to evaluate response reliability.",
            "pros": ["High customizability", "Multiple perspectives", "Flexible prompting"],
            "cons": ["Variable cost", "Judge selection matters", "Potential bias"],
            "scorers": ["Categorical Judge", "Continuous Judge", "Panel Average", "Likert Scale"]
        },
        "UQ Ensemble": {
            "description": "Combines multiple uncertainty quantification methods.",
            "pros": ["Robust estimates", "Best of all methods", "Tunable weights"],
            "cons": ["Complex setup", "Higher computation", "Method dependencies"],
            "scorers": ["Weighted Average", "BS Detector", "Generalized UQ Ensemble"]
        },
        "Semantic Entropy": {
            "description": "Combines token probabilities with semantic clustering.",
            "pros": ["Semantic awareness", "Probability-based", "Clustering insights"],
            "cons": ["Complex computation", "Parameter tuning", "Model-specific"],
            "scorers": ["Semantic Entropy", "Token Entropy", "Cluster-based Metrics"]
        }
    }
    
    for method, info in method_info.items():
        with st.expander(f"📖 {method}"):
            st.write(f"**Description:** {info['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Pros:**")
                for pro in info['pros']:
                    st.write(f"• {pro}")
            
            with col2:
                st.write("**Cons:**")
                for con in info['cons']:
                    st.write(f"• {con}")
            
            st.write("**Available Scorers:**")
            for scorer in info['scorers']:
                st.write(f"• {scorer}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **UQLM Demo Application** | Built with Streamlit and UQLM  
    For more information, visit the [UQLM GitHub repository](https://github.com/cvs-health/uqlm)
    """)

if __name__ == "__main__":
    main()