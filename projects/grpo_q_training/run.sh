#!/bin/bash

# GRPO Mathematical Reasoning Training Runner
# This script provides an easy way to run the GRPO training pipeline

set -e  # Exit on any error

echo "🚀 GRPO Mathematical Reasoning Training"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "grpo_training.py" ]; then
    echo "❌ Error: grpo_training.py not found. Please run this script from the grpo_q_training directory."
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  test      Run setup tests to verify installation"
    echo "  install   Install required dependencies"
    echo "  train     Run GRPO training (default)"
    echo "  help      Show this help message"
    echo ""
    echo "Training Options (when using 'train' command):"
    echo "  --quick       Quick training (10 steps, 100 examples)"
    echo "  --medium      Medium training (50 steps, 1000 examples) [default]"
    echo "  --full        Full training (200 steps, all examples)"
    echo "  --custom      Use custom parameters (will prompt for input)"
    echo ""
    echo "Examples:"
    echo "  $0 test                    # Test setup"
    echo "  $0 install                 # Install dependencies"
    echo "  $0 train                   # Run default training"
    echo "  $0 train --quick           # Quick training"
    echo "  $0 train --full            # Full training"
}

# Function to install dependencies
install_deps() {
    echo "📦 Installing dependencies..."
    
    if command -v pip &> /dev/null; then
        pip install -r requirements.txt
        echo "✅ Dependencies installed successfully!"
    else
        echo "❌ Error: pip not found. Please install pip first."
        exit 1
    fi
}

# Function to run tests
run_tests() {
    echo "🔍 Running setup tests..."
    python test_setup.py
}

# Function to run training with different configurations
run_training() {
    local config="$1"
    
    case $config in
        "quick")
            echo "🏃 Running quick training (10 steps, 100 examples)..."
            python run_training.py \
                --max-steps 10 \
                --max-examples 100 \
                --batch-size 1 \
                --gradient-accumulation-steps 4
            ;;
        "medium")
            echo "🚶 Running medium training (50 steps, 1000 examples)..."
            python run_training.py \
                --max-steps 50 \
                --max-examples 1000 \
                --batch-size 2 \
                --gradient-accumulation-steps 8
            ;;
        "full")
            echo "🏋️ Running full training (200 steps, all examples)..."
            python run_training.py \
                --max-steps 200 \
                --max-examples 0 \
                --batch-size 4 \
                --gradient-accumulation-steps 4
            ;;
        "custom")
            echo "⚙️ Custom training configuration..."
            echo "Please enter your training parameters:"
            
            read -p "Max steps (default: 50): " max_steps
            max_steps=${max_steps:-50}
            
            read -p "Max examples (default: 1000, 0 for all): " max_examples
            max_examples=${max_examples:-1000}
            
            read -p "Batch size (default: 2): " batch_size
            batch_size=${batch_size:-2}
            
            read -p "Learning rate (default: 5e-6): " learning_rate
            learning_rate=${learning_rate:-5e-6}
            
            echo "Running custom training..."
            python run_training.py \
                --max-steps $max_steps \
                --max-examples $max_examples \
                --batch-size $batch_size \
                --learning-rate $learning_rate
            ;;
        *)
            echo "🚶 Running default training (medium configuration)..."
            python run_training.py
            ;;
    esac
}

# Main script logic
case "${1:-train}" in
    "help"|"-h"|"--help")
        show_usage
        ;;
    "install")
        install_deps
        ;;
    "test")
        run_tests
        ;;
    "train")
        # Check for training configuration
        config="${2:-medium}"
        if [[ "$config" == --* ]]; then
            config="${config#--}"
        fi
        run_training "$config"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo "✅ Script completed successfully!"