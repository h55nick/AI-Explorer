#!/usr/bin/env python3
"""
Configuration file for GRPO training with predefined settings
"""

from grpo_training import GRPOTrainingConfig


def get_quick_config():
    """Quick training configuration for testing"""
    config = GRPOTrainingConfig()
    config.max_steps = 10
    config.per_device_train_batch_size = 1
    config.gradient_accumulation_steps = 4
    config.learning_rate = 1e-5
    config.output_dir = "./quick_grpo_outputs"
    return config


def get_medium_config():
    """Medium training configuration (default)"""
    config = GRPOTrainingConfig()
    config.max_steps = 50
    config.per_device_train_batch_size = 2
    config.gradient_accumulation_steps = 8
    config.learning_rate = 5e-6
    config.output_dir = "./medium_grpo_outputs"
    return config


def get_full_config():
    """Full training configuration for best results"""
    config = GRPOTrainingConfig()
    config.max_steps = 200
    config.per_device_train_batch_size = 4
    config.gradient_accumulation_steps = 4
    config.learning_rate = 3e-6
    config.output_dir = "./full_grpo_outputs"
    config.save_steps = 25
    return config


def get_gpu_optimized_config():
    """Configuration optimized for high-end GPUs"""
    config = GRPOTrainingConfig()
    config.max_steps = 100
    config.per_device_train_batch_size = 8
    config.gradient_accumulation_steps = 2
    config.learning_rate = 5e-6
    config.max_prompt_length = 1536
    config.max_completion_length = 1536
    config.output_dir = "./gpu_optimized_grpo_outputs"
    return config


def get_cpu_config():
    """Configuration for CPU-only training (very slow)"""
    config = GRPOTrainingConfig()
    config.max_steps = 5
    config.per_device_train_batch_size = 1
    config.gradient_accumulation_steps = 2
    config.learning_rate = 1e-4
    config.max_prompt_length = 512
    config.max_completion_length = 512
    config.output_dir = "./cpu_grpo_outputs"
    config.use_trackio = False  # Disable tracking for CPU
    return config


def get_debug_config():
    """Configuration for debugging and development"""
    config = GRPOTrainingConfig()
    config.max_steps = 3
    config.per_device_train_batch_size = 1
    config.gradient_accumulation_steps = 1
    config.learning_rate = 1e-4
    config.logging_steps = 1
    config.output_dir = "./debug_grpo_outputs"
    config.use_trackio = False
    return config


# Configuration registry
CONFIGS = {
    'quick': get_quick_config,
    'medium': get_medium_config,
    'full': get_full_config,
    'gpu_optimized': get_gpu_optimized_config,
    'cpu': get_cpu_config,
    'debug': get_debug_config,
}


def get_config(config_name='medium'):
    """Get a predefined configuration by name"""
    if config_name not in CONFIGS:
        available = ', '.join(CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return CONFIGS[config_name]()


def list_configs():
    """List all available configurations"""
    print("Available configurations:")
    for name, func in CONFIGS.items():
        config = func()
        print(f"  {name:15} - {config.max_steps:3} steps, batch {config.per_device_train_batch_size}, lr {config.learning_rate}")


if __name__ == "__main__":
    list_configs()