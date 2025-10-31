#!/usr/bin/env python3
"""
Simple script to run GRPO training with customizable parameters
"""

import argparse
import sys
from grpo_training import GRPOTrainingConfig, GRPOTrainingPipeline, ModelEvaluator


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run GRPO Mathematical Reasoning Training")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                       help="Model name to use for training")
    
    # Training configuration
    parser.add_argument("--max-steps", type=int, default=50,
                       help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=5e-6,
                       help="Learning rate for training")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Per device training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                       help="Gradient accumulation steps")
    
    # Dataset configuration
    parser.add_argument("--max-examples", type=int, default=1000,
                       help="Maximum number of training examples to use (None for all)")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./trl_grpo_outputs",
                       help="Output directory for model checkpoints")
    
    # Experiment tracking
    parser.add_argument("--no-trackio", action="store_true",
                       help="Disable trackio experiment tracking")
    parser.add_argument("--project-name", type=str, default="GRPO-Mathematical-Reasoning",
                       help="Project name for experiment tracking")
    
    # Evaluation
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip model evaluation after training")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create configuration
    config = GRPOTrainingConfig()
    
    # Update configuration with command line arguments
    config.model_name = args.model_name
    config.max_steps = args.max_steps
    config.learning_rate = args.learning_rate
    config.per_device_train_batch_size = args.batch_size
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.output_dir = args.output_dir
    config.use_trackio = not args.no_trackio
    config.project_name = args.project_name
    
    print(f"🚀 Starting GRPO Training with configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Max steps: {config.max_steps}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Batch size: {config.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Max examples: {args.max_examples}")
    print(f"   Output dir: {config.output_dir}")
    print(f"   Trackio: {'Enabled' if config.use_trackio else 'Disabled'}")
    
    # Create and run training pipeline
    pipeline = GRPOTrainingPipeline(config)
    
    try:
        # Train the model
        model, tokenizer, trainer = pipeline.train(max_examples=args.max_examples)
        
        if not args.skip_evaluation:
            # Create evaluator
            evaluator = ModelEvaluator(model, tokenizer, pipeline.reward_functions)
            
            # Test with sample questions
            test_questions = [
                {
                    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                    "answer": "72"
                },
                {
                    "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
                    "answer": "10"
                },
                {
                    "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents decided to give her twice as much as her parents. How much more money does Betty need to buy the wallet?",
                    "answer": "5"
                }
            ]
            
            # Run evaluation
            evaluation_results = evaluator.run_evaluation_suite(test_questions)
            
            print(f"\n📊 Final Results:")
            print(f"   Format compliance: {evaluation_results['format_compliance_rate']:.2%}")
            print(f"   Accuracy: {evaluation_results['accuracy_rate']:.2%}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"💾 Model saved to: {config.output_dir}")
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up resources
        pipeline.cleanup_resources()


if __name__ == "__main__":
    main()