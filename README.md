# UQLM - Uncertainty Quantification for Language Models

A comprehensive Streamlit application for testing and demonstrating various uncertainty quantification methods for language models using Google's Gemini API.

## Features

- **Multiple UQ Methods**: BlackBox UQ, WhiteBox UQ, Semantic Entropy, LLM Panel, and UQ Ensemble
- **Model Selection**: Support for multiple Gemini models (gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro, gemini-1.0-pro)
- **Interactive Interface**: Easy-to-use Streamlit web interface
- **Real-time Results**: Live uncertainty quantification with detailed metrics
- **Multiple Input Methods**: Single prompt, multiple prompts, or example prompts

## Setup

### Prerequisites

1. **Python 3.8+**
2. **Google API Key** - Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r projects/uc_exploration/requirements.txt
   ```

### Dependencies

The application requires the following packages:
- `streamlit` - Web interface
- `uqlm` - Uncertainty quantification methods
- `google-generativeai>=1.47.0` - Google Gemini API
- `langchain-google-genai` - LangChain integration
- `langchain-core` - LangChain core functionality

## Usage

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run projects/uc_exploration/uqlm_app.py
   ```

2. Open your browser to `http://localhost:8501`

3. **Enter your Google API Key** in the sidebar configuration section

4. **Select a Gemini model** from the dropdown (default: gemini-2.0-flash)

5. **Choose UQ methods** to run (BlackBox UQ, WhiteBox UQ, Semantic Entropy, etc.)

6. **Enter your prompt(s)** and click "Run Uncertainty Quantification"

### Available UQ Methods

#### 1. BlackBox UQ
- Generates multiple responses and analyzes their consistency
- Uses semantic similarity, exact matching, and contradiction detection
- Configurable number of responses (3-10)

#### 2. WhiteBox UQ  
- Analyzes token-level probabilities from the model
- Requires log probabilities support (implemented for Gemini)
- Provides detailed uncertainty metrics

#### 3. Semantic Entropy
- Measures semantic uncertainty across multiple generations
- Groups semantically similar responses
- Calculates entropy over semantic clusters

#### 4. LLM Panel
- Uses multiple model calls with different parameters
- Analyzes consistency across different sampling strategies
- Provides ensemble-based uncertainty estimates

#### 5. UQ Ensemble
- Combines multiple uncertainty quantification methods
- Provides aggregated uncertainty scores
- Offers comprehensive uncertainty analysis

### Input Methods

1. **Single Prompt**: Enter one prompt for analysis
2. **Multiple Prompts**: Enter multiple prompts (one per line)
3. **Example Prompts**: Use pre-defined example prompts for testing

### Example Prompts

Try these example prompts to test the system:

- "What are the main causes of climate change?"
- "Explain quantum computing in simple terms."
- "What will be the impact of AI on society?"
- "How does photosynthesis work?"
- "What are the benefits of renewable energy?"

## API Key Setup

### Option 1: Through the Web Interface
1. Enter your Google API key in the sidebar
2. The key is stored in your browser session

### Option 2: Environment Variable
```bash
export GOOGLE_API_KEY="your_api_key_here"
streamlit run projects/uc_exploration/uqlm_app.py
```

### Getting a Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key
5. Paste it into the application

## Understanding the Results

### Uncertainty Metrics

- **Semantic Negentropy**: Measures semantic consistency (higher = more consistent)
- **Exact Match**: Percentage of identical responses
- **Non-contradiction**: Measures logical consistency
- **BERT Score**: Semantic similarity using BERT embeddings
- **Min Probability**: Lowest token probability in the response
- **Length Normalized**: Uncertainty normalized by response length

### Interpreting Scores

- **High Uncertainty**: Model is unsure about the answer
- **Low Uncertainty**: Model is confident in the response
- **Semantic Entropy**: Higher values indicate more semantic diversity
- **Probability Scores**: Lower values indicate higher uncertainty

## Troubleshooting

### Common Issues

1. **API Key Invalid**: 
   - Verify your API key is correct
   - Check if the key has proper permissions
   - Try regenerating the key

2. **Event Loop Errors**:
   - The app handles async operations automatically
   - Restart the app if you encounter persistent errors

3. **Model Not Available**:
   - Some models may not be available in your region
   - Try switching to a different model (e.g., gemini-1.5-flash)

4. **Rate Limiting**:
   - Google API has rate limits
   - Wait a moment between requests if you hit limits

### Performance Tips

- Start with fewer responses for BlackBox UQ (3-5) for faster results
- Use simpler prompts for initial testing
- WhiteBox UQ is generally faster than BlackBox UQ
- Semantic Entropy may take longer due to clustering operations

## Technical Details

### Architecture

- **Frontend**: Streamlit web interface
- **Backend**: UQLM library for uncertainty quantification
- **API**: Google Gemini API for language model access
- **Integration**: LangChain for model abstraction

### Supported Models

- `gemini-2.0-flash` (default) - Latest and fastest
- `gemini-1.5-flash` - Fast and efficient
- `gemini-1.5-pro` - More capable, slower
- `gemini-1.0-pro` - Original Gemini model

### Custom Features

- **Log Probabilities**: Custom implementation for WhiteBox UQ
- **Async Handling**: Proper event loop management for Streamlit
- **Error Recovery**: Graceful handling of API errors
- **Session Management**: Persistent configuration across sessions

## Contributing

This application is built on top of the [UQLM library](https://github.com/cvs-health/uqlm). For issues or contributions related to the core uncertainty quantification methods, please refer to the main UQLM repository.

## License

This application follows the same license as the UQLM library. Please refer to the [UQLM GitHub repository](https://github.com/cvs-health/uqlm) for license details.

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Verify your API key and internet connection
3. Try with different models or simpler prompts
4. Refer to the [UQLM documentation](https://github.com/cvs-health/uqlm)

---

**Built with ❤️ using Streamlit and UQLM**