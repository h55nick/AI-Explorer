# GRPO Training Usage Guide

## Quick Start

### 1. Test Your Setup
```bash
python test_setup.py
```

### 2. Run Training (Multiple Options)

#### Option A: Using the shell script (Recommended)
```bash
# Quick test (10 steps, 100 examples)
./run.sh train --quick

# Medium training (50 steps, 1000 examples) - Default
./run.sh train --medium

# Full training (200 steps, all examples)
./run.sh train --full

# Custom configuration
./run.sh train --custom
```

#### Option B: Using Python directly
```bash
# Basic training
python run_training.py

# Custom parameters
python run_training.py \
    --max-steps 100 \
    --learning-rate 1e-5 \
    --batch-size 4 \
    --max-examples 2000 \
    --output-dir ./my_grpo_model
```

#### Option C: Using the main script directly
```bash
python grpo_training.py
```

### 3. Monitor Training

If trackio is enabled (default), you can monitor training progress:
```python
import trackio
trackio.show(project="GRPO-Mathematical-Reasoning")
```

## Configuration Options

### Pre-defined Configurations
```python
from config import get_config

# Available configurations:
config = get_config('quick')      # Fast testing
config = get_config('medium')     # Default balanced
config = get_config('full')       # Best results
config = get_config('gpu_optimized')  # For high-end GPUs
config = get_config('cpu')        # CPU-only training
config = get_config('debug')      # Development/debugging
```

### Custom Configuration
```python
from grpo_training import GRPOTrainingConfig, GRPOTrainingPipeline

config = GRPOTrainingConfig()
config.max_steps = 200
config.learning_rate = 1e-5
config.model_name = "microsoft/DialoGPT-medium"

pipeline = GRPOTrainingPipeline(config)
model, tokenizer, trainer = pipeline.train(max_examples=5000)
```

## Expected Output

The training will produce:
1. **Model checkpoints** in the output directory
2. **Training logs** with reward scores and metrics
3. **Evaluation results** on sample problems
4. **Trackio dashboard** (if enabled) for monitoring

### Sample Training Output
```
🚀 Starting GRPO Mathematical Reasoning Training Pipeline
📊 Model parameters: ~3000.0M
📊 LoRA Training Parameters Summary:
trainable params: 8,388,608 || all params: 3,008,388,608 || trainable%: 0.2788

✅ GRPO Trainer initialized successfully!
📊 Training dataset: 1,000 examples
🎯 Reward functions: 4 active

🚀 Starting GRPO training...
Step 1/50: reward_0=0.2, reward_1=0.5, reward_2=0.1, reward_3=0.3
...
✅ Training completed successfully!
💾 Model saved to: ./trl_grpo_outputs

📊 Evaluation Summary:
   Format compliance: 85.00%
   Accuracy: 66.67%
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use CPU config
2. **Slow Training**: Use GPU environment or reduce dataset size
3. **Import Errors**: Run `pip install -r requirements.txt`
4. **Poor Results**: Increase training steps or use larger model

### Performance Tips

- **For faster training**: Use GPU, reduce max_examples
- **For better results**: Increase max_steps, use larger model
- **For debugging**: Use debug config with minimal steps

## File Structure After Training

```
grpo_q_training/
├── grpo_training.py          # Main training script
├── run_training.py           # CLI runner
├── test_setup.py            # Setup verification
├── config.py                # Pre-defined configurations
├── run.sh                   # Shell script runner
├── requirements.txt         # Dependencies
├── README.md               # Full documentation
├── USAGE.md                # This usage guide
└── trl_grpo_outputs/       # Training outputs (created after training)
    ├── checkpoint-25/
    ├── checkpoint-50/
    └── pytorch_model.bin
```

## Next Steps

After training, you can:
1. **Evaluate** the model on more test cases
2. **Fine-tune** with different hyperparameters
3. **Deploy** the model for inference
4. **Compare** different reward function combinations

For detailed documentation, see `README.md`.