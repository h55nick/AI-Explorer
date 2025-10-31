# GRPO Mathematical Reasoning Training

This project implements **Advanced GRPO (Group Relative Policy Optimization)** for mathematical reasoning using a comprehensive multi-reward training system. The implementation is based on the [HuggingFace Cookbook tutorial](https://huggingface.co/learn/cookbook/en/trl_grpo_reasoning_advanced_reward) and fine-tunes models on the GSM8K dataset with four specialized reward functions.

## Key Features

- **4 Reward Functions**: Format compliance, approximate matching, answer correctness, and number extraction
- **Memory Efficient**: 4-bit quantization + LoRA for consumer GPUs
- **Interactive Monitoring**: Real-time training metrics with trackio dashboard
- **Structured Output**: Enforces step-by-step reasoning format
- **Modular Design**: Clean, extensible codebase with separate components

## Project Structure

```
grpo_q_training/
├── grpo_training.py      # Main training pipeline implementation
├── run_training.py       # Simple script to run training with CLI arguments
├── requirements.txt      # Python dependencies
└── README.md            # This documentation
```

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd AI-Explorer/projects/grpo_q_training
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU availability** (recommended):
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU name: {torch.cuda.get_device_name()}")
   ```

## Quick Start

### Basic Training

Run training with default parameters:

```bash
python run_training.py
```

### Customized Training

Run training with custom parameters:

```bash
python run_training.py \
    --max-steps 100 \
    --learning-rate 1e-5 \
    --batch-size 4 \
    --max-examples 2000 \
    --output-dir ./my_grpo_model
```

### Advanced Usage

For more control, you can modify the configuration directly in Python:

```python
from grpo_training import GRPOTrainingConfig, GRPOTrainingPipeline

# Create custom configuration
config = GRPOTrainingConfig()
config.max_steps = 200
config.learning_rate = 1e-5
config.model_name = "microsoft/DialoGPT-medium"  # Use different model

# Run training
pipeline = GRPOTrainingPipeline(config)
model, tokenizer, trainer = pipeline.train(max_examples=5000)
```

## Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `Qwen/Qwen2.5-3B-Instruct` | Model name to use for training |
| `--max-steps` | `50` | Maximum training steps |
| `--learning-rate` | `5e-6` | Learning rate for training |
| `--batch-size` | `2` | Per device training batch size |
| `--gradient-accumulation-steps` | `8` | Gradient accumulation steps |
| `--max-examples` | `1000` | Maximum number of training examples |
| `--output-dir` | `./trl_grpo_outputs` | Output directory for model checkpoints |
| `--no-trackio` | `False` | Disable trackio experiment tracking |
| `--project-name` | `GRPO-Mathematical-Reasoning` | Project name for tracking |
| `--skip-evaluation` | `False` | Skip model evaluation after training |

### Configuration Class Parameters

The `GRPOTrainingConfig` class provides fine-grained control over all training parameters:

```python
class GRPOTrainingConfig:
    # Model configuration
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    max_seq_length = 2048
    
    # Training configuration
    learning_rate = 5e-6
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    max_steps = 100
    
    # LoRA configuration
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = ["q_proj", "v_proj"]
    
    # Generation configuration
    max_prompt_length = 1024
    max_completion_length = 1024
    num_generations = 4
```

## Reward Functions

The training uses four specialized reward functions to evaluate different aspects of mathematical reasoning:

### 1. Exact Format Matching (`match_format_exactly`)
- **Score**: 3.0 for perfect format adherence, 0.0 otherwise
- **Purpose**: Ensures model learns complete structured output pattern
- **Checks**: Presence of reasoning and solution sections in correct format

### 2. Approximate Format Matching (`match_format_approximately`)
- **Score**: Graduated scoring for individual format elements
- **Purpose**: Encourages learning components even if not perfect
- **Scoring**: +0.5 for each correct element, -0.5 for missing/incorrect

### 3. Answer Correctness (`check_answer_correctness`)
- **Score**: 
  - 3.0: Exact match
  - 1.5: Within 10% (close answer)
  - 0.5: Within 20% (reasonable attempt)
  - -0.5: Wrong answer (penalty)
- **Purpose**: Graduated scoring for mathematical accuracy

### 4. Number Extraction (`check_numbers_extraction`)
- **Score**: 1.5 for correct extraction, 0.0 otherwise
- **Purpose**: Tests ability to extract numerical values from solution sections

## Output Format

The model is trained to generate responses in this structured format:

```
<start_working_out>
Step 1: [reasoning step]
Step 2: [reasoning step]
...
<end_working_out>

<SOLUTION>
[final numerical answer]
</SOLUTION>
```

## Memory Requirements

The implementation is optimized for consumer GPUs:

- **4-bit quantization**: ~75% memory reduction vs FP16
- **LoRA**: Trains only ~0.1% of parameters
- **Recommended**: 8GB+ GPU memory
- **Minimum**: 6GB GPU memory (with reduced batch size)

## Experiment Tracking

The pipeline includes optional experiment tracking with trackio:

1. **Enable tracking** (default):
   ```bash
   python run_training.py  # trackio enabled by default
   ```

2. **Disable tracking**:
   ```bash
   python run_training.py --no-trackio
   ```

3. **View experiment dashboard**:
   ```python
   import trackio
   trackio.show(project="GRPO-Mathematical-Reasoning")
   ```

## Evaluation

After training, the model is automatically evaluated on sample GSM8K problems. The evaluation includes:

- **Format Compliance**: Percentage of responses following the structured format
- **Mathematical Accuracy**: Percentage of correct numerical answers
- **Detailed Analysis**: Per-question breakdown of performance

### Sample Evaluation Output

```
📊 Evaluation Summary:
   Total questions: 3
   Format compliance: 3/3 (100.00%)
   Accuracy: 2/3 (66.67%)

📝 Sample Evaluation:
Question: Natalia sold clips to 48 of her friends in April...
Expected: 72
Model Response:
<start_working_out>
Natalia sold 48 clips in April.
In May, she sold half as many, so 48 ÷ 2 = 24 clips.
Total clips = 48 + 24 = 72 clips.
<end_working_out>

<SOLUTION>
72
</SOLUTION>
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `per_device_train_batch_size` to 1
   - Reduce `max_examples` to 500
   - Ensure no other GPU processes are running

2. **Slow Training**:
   - Increase `gradient_accumulation_steps` to maintain effective batch size
   - Use smaller model (e.g., `microsoft/DialoGPT-small`)
   - Reduce `max_completion_length`

3. **Poor Format Compliance**:
   - Increase `max_steps` for more training
   - Adjust reward function weights
   - Check system prompt formatting

4. **Installation Issues**:
   ```bash
   # Install specific versions if needed
   pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
   pip install transformers==4.36.0
   ```

### Performance Optimization

1. **For faster training**:
   - Use `xformers` for memory-efficient attention
   - Enable `torch.compile()` for PyTorch 2.0+
   - Use mixed precision training

2. **For better results**:
   - Increase `max_steps` to 200-500
   - Use larger models (e.g., 7B parameters)
   - Increase `max_examples` to use full dataset

## Advanced Usage

### Custom Reward Functions

You can add custom reward functions by extending the `RewardFunctions` class:

```python
class CustomRewardFunctions(RewardFunctions):
    def custom_reward(self, prompts, completions, answer, **kwargs):
        # Your custom reward logic here
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            # Calculate score based on your criteria
            score = calculate_custom_score(response)
            scores.append(score)
        return scores
```

### Different Models

The pipeline supports various models:

```python
# Smaller models (faster training)
config.model_name = "microsoft/DialoGPT-small"
config.model_name = "distilgpt2"

# Larger models (better performance)
config.model_name = "Qwen/Qwen2.5-7B-Instruct"
config.model_name = "microsoft/DialoGPT-large"
```

### Custom Datasets

To use different datasets, modify the `DatasetProcessor` class:

```python
def load_custom_dataset(self):
    # Load your custom dataset
    dataset = load_dataset("your/dataset")
    # Process according to your format
    return processed_dataset
```

## References

- **Original Tutorial**: [HuggingFace GRPO Cookbook](https://huggingface.co/learn/cookbook/en/trl_grpo_reasoning_advanced_reward)
- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- **GSM8K Dataset**: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- **LoRA**: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **TRL Library**: [Transformers Reinforcement Learning](https://github.com/huggingface/trl)

## License

This project follows the same license as the original HuggingFace cookbook tutorial.

## Contributing

Feel free to submit issues and enhancement requests!

## Support

For questions and support:
1. Check the troubleshooting section above
2. Review the original HuggingFace tutorial
3. Open an issue in the repository