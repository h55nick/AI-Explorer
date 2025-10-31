# 🚀 UQLM Quick Start Guide

## What You Have

A complete **UQLM (Uncertainty Quantification for Language Models)** application with:

- ✅ **Streamlit Web Interface** - Interactive web app for testing UQ methods
- ✅ **Multiple Gemini Models** - Support for gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro, gemini-1.0-pro
- ✅ **All UQ Methods** - BlackBox UQ, WhiteBox UQ, Semantic Entropy, LLM Panel, UQ Ensemble
- ✅ **Real Log Probabilities** - Custom implementation for WhiteBox UQ with actual token probabilities
- ✅ **No Demo Mode** - Requires real API key, no fallback modes
- ✅ **Event Loop Fixed** - Proper async handling for Streamlit

## 🎯 Getting Started

To run the UQLM application:

```bash
streamlit run projects/uc_exploration/uqlm_app.py
```

The app will be available at `http://localhost:8501`

## 🔑 What You Need

1. **Google API Key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Enter the API key in the sidebar when you visit the app

## 🧪 How to Test

### Option 1: Web Interface (Recommended)
1. Run `streamlit run projects/uc_exploration/uqlm_app.py`
2. Visit `http://localhost:8501`
3. Enter your Google API Key in the sidebar
4. Select a Gemini model (default: gemini-2.0-flash)
5. Choose UQ methods to test
6. Enter a prompt like: "What are the main causes of climate change?"
7. Click "Run Uncertainty Quantification"

### Option 2: CLI Test Script
```bash
cd /workspace/project/AI-Explorer
python projects/uc_exploration/uqlm_cli_demo.py
```
This will test BlackBox UQ, WhiteBox UQ, and Semantic Entropy methods.

## 📁 Key Files

- **`projects/uc_exploration/uqlm_app.py`** - Main Streamlit application
- **`projects/uc_exploration/gemini_whitebox_llm.py`** - Custom LLM with log probabilities for WhiteBox UQ
- **`projects/uc_exploration/uqlm_cli_demo.py`** - CLI testing script
- **`projects/uc_exploration/requirements.txt`** - All dependencies
- **`README.md`** - Comprehensive documentation

## 🎯 Available UQ Methods

1. **BlackBox UQ** - Consistency-based uncertainty using multiple responses
2. **WhiteBox UQ** - Token probability-based uncertainty (with real log probabilities!)
3. **Semantic Entropy** - Semantic clustering-based uncertainty
4. **LLM Panel** - Multiple model comparison
5. **UQ Ensemble** - Combined uncertainty methods

## 🔧 Technical Features

- **Model Selection**: Choose from 4 Gemini models
- **Real Log Probabilities**: Custom implementation bypasses LangChain limitations
- **Async Support**: Proper event loop handling for Streamlit
- **Error Recovery**: Graceful handling of API errors
- **No Demo Mode**: Enforces real API usage

## 🚨 Important Notes

1. **API Key Required**: The app will NOT work without a valid Google API key
2. **No Demo Fallbacks**: All demo modes have been removed as requested
3. **Rate Limits**: Google API has rate limits - wait between requests if needed
4. **Model Availability**: Some models may not be available in all regions

## 🎉 Ready to Use!

Your UQLM application is fully functional and ready for testing all uncertainty quantification methods with Google's Gemini models. The app enforces real API usage and provides accurate uncertainty measurements.

**Next Steps:**
1. Get your Google API key
2. Visit the web app
3. Start testing with your prompts!

---
**Built with ❤️ using Streamlit, UQLM, and Google Gemini**