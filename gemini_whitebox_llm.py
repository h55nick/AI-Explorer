#!/usr/bin/env python3
"""
Custom Gemini LLM wrapper with log probabilities support for UQLM WhiteBox UQ.
This creates a simple wrapper that UQLM can use directly.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union
import google.genai as genai
from google.genai import types


class GeminiWhiteBoxLLM:
    """Simple Gemini LLM with log probabilities support for WhiteBox UQ."""
    
    def __init__(self, 
                 model_name: str = "gemini-2.0-flash",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure the API
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key is required")
        
        # Initialize the client with the new google-genai SDK
        self.client = genai.Client(api_key=api_key)
        
        # Set the logprobs attribute that UQLM expects
        self.logprobs = True
        
        # Generation config with logprobs enabled
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_logprobs=True,  # Enable log probabilities
            logprobs=5  # Return top 5 alternative tokens
        )
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate response with log probabilities."""
        try:
            # Generate content with logprobs using the new SDK
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generation_config
            )
            
            # Extract the response text
            response_text = ""
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        response_text = candidate.content.parts[0].text
            
            # Extract log probabilities if available
            logprobs_data = None
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                print(f"Debug: candidate attributes: {[attr for attr in dir(candidate) if not attr.startswith('_')]}")
                if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                    print(f"Debug: Found logprobs_result")
                    logprobs_data = self._extract_logprobs(candidate.logprobs_result)
                else:
                    print(f"Debug: No logprobs_result found")
            
            return {
                'text': response_text,
                'logprobs': logprobs_data,
                'finish_reason': getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None
            }
            
        except Exception as e:
            raise RuntimeError(f"Error generating content with logprobs: {e}")
    
    def _extract_logprobs(self, logprobs_result) -> Dict[str, Any]:
        """Extract log probabilities from the response."""
        try:
            logprobs_data = {
                'token_logprobs': [],
                'top_logprobs': [],
                'tokens': []
            }
            
            # Debug: print the structure to understand it
            print(f"Debug: logprobs_result type: {type(logprobs_result)}")
            print(f"Debug: logprobs_result attributes: {[attr for attr in dir(logprobs_result) if not attr.startswith('_')]}")
            
            # Extract chosen candidates (the actual tokens used)
            if hasattr(logprobs_result, 'chosen_candidates') and logprobs_result.chosen_candidates:
                for token_data in logprobs_result.chosen_candidates:
                    if hasattr(token_data, 'token'):
                        logprobs_data['tokens'].append(token_data.token)
                    if hasattr(token_data, 'log_probability'):
                        logprobs_data['token_logprobs'].append(token_data.log_probability)
            
            # Extract top candidates (alternatives for each position)
            if hasattr(logprobs_result, 'top_candidates') and logprobs_result.top_candidates:
                for position_data in logprobs_result.top_candidates:
                    if hasattr(position_data, 'candidates'):
                        top_alternatives = []
                        for alt in position_data.candidates:
                            if hasattr(alt, 'token') and hasattr(alt, 'log_probability'):
                                top_alternatives.append({
                                    'token': alt.token,
                                    'logprob': alt.log_probability
                                })
                        logprobs_data['top_logprobs'].append(top_alternatives)
            
            print(f"Debug: Extracted {len(logprobs_data['tokens'])} tokens with logprobs")
            
            return logprobs_data
            
        except Exception as e:
            print(f"Warning: Could not extract logprobs: {e}")
            return {'token_logprobs': [], 'top_logprobs': [], 'tokens': []}
    
    def get_token_ids(self, text: str) -> List[int]:
        """Get token IDs for the given text (placeholder implementation)."""
        # This would need to be implemented with the actual tokenizer
        # For now, return a simple approximation
        return list(range(len(text.split())))


def create_gemini_whitebox_llm(api_key: Optional[str] = None, 
                               model_name: str = "gemini-2.0-flash",
                               **kwargs) -> GeminiWhiteBoxLLM:
    """Factory function to create a Gemini LLM with logprobs support."""
    return GeminiWhiteBoxLLM(
        model_name=model_name,
        api_key=api_key,
        **kwargs
    )


# Test function
async def test_whitebox_llm():
    """Test the whitebox LLM implementation."""
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("❌ No API key found. Set GOOGLE_API_KEY environment variable.")
            return False
        
        print("🧪 Testing Gemini WhiteBox LLM with logprobs...")
        
        llm = create_gemini_whitebox_llm(api_key=api_key)
        
        # Test generation
        result = llm.generate("What is the capital of France?")
        
        print(f"✅ Response: {result['text']}")
        
        if result['logprobs']:
            logprobs = result['logprobs']
            print(f"✅ Logprobs extracted: {len(logprobs.get('tokens', []))} tokens")
            print(f"   Sample tokens: {logprobs.get('tokens', [])[:5]}")
            print(f"   Sample logprobs: {logprobs.get('token_logprobs', [])[:5]}")
            return True
        else:
            print("⚠️ No logprobs found in response")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_whitebox_llm())