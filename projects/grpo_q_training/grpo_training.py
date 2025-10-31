#!/usr/bin/env python3
"""
Advanced GRPO Fine-tuning for Mathematical Reasoning with Multi-Reward Training

This script implements the complete GRPO (Group Relative Policy Optimization) pipeline
for training mathematical reasoning models on the GSM8K dataset with four specialized
reward functions.

Key Features:
- 4 Reward Functions: Format compliance, approximate matching, answer correctness, and number extraction
- Memory Efficient: 4-bit quantization + LoRA for consumer GPUs
- Interactive Monitoring: Real-time training metrics with trackio dashboard
- Structured Output: Enforces step-by-step reasoning format

Based on: https://huggingface.co/learn/cookbook/en/trl_grpo_reasoning_advanced_reward
"""

import os
import re
import gc
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import trackio
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("gradio_client").setLevel(logging.WARNING)


class GRPOTrainingConfig:
    """Configuration class for GRPO training parameters"""
    
    def __init__(self):
        # Model configuration
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.max_seq_length = 2048
        
        # Training configuration
        self.learning_rate = 5e-6
        self.per_device_train_batch_size = 2
        self.gradient_accumulation_steps = 8
        self.max_steps = 100  # Increased from 10 for better training
        self.logging_steps = 1
        self.max_grad_norm = 0.1
        
        # Generation configuration
        self.max_prompt_length = 1024
        self.max_completion_length = 1024
        self.num_generations = 4  # Default GRPO generations
        
        # LoRA configuration
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = ["q_proj", "v_proj"]
        
        # Output configuration
        self.output_dir = "./trl_grpo_outputs"
        self.save_steps = 50
        
        # Experiment tracking
        self.project_name = "GRPO-Mathematical-Reasoning"
        self.use_trackio = True


class RewardFunctions:
    """Container for all reward functions used in GRPO training"""
    
    def __init__(self):
        # Define format tokens
        self.reasoning_start = "<start_working_out>"
        self.reasoning_end = "<end_working_out>"
        self.solution_start = "<SOLUTION>"
        self.solution_end = "</SOLUTION>"
        
        # Compile regex patterns
        self.match_format = re.compile(
            rf"^[\s]{{0,}}"
            rf"{self.reasoning_start}.+?{self.reasoning_end}.*?"
            rf"{self.solution_start}(.+?){self.solution_end}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL
        )
        
        self.match_numbers = re.compile(
            rf"{self.solution_start}.*?([\d\.]{{1,}})",
            flags=re.MULTILINE | re.DOTALL
        )
    
    def match_format_exactly(self, completions, **kwargs):
        """
        High reward (3.0) for perfect format adherence
        Ensures model learns the complete structured output pattern
        """
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            score = 3.0 if self.match_format.search(response) is not None else 0.0
            scores.append(score)
        return scores
    
    def match_format_approximately(self, completions, **kwargs):
        """
        Graduated scoring for format elements
        Encourages learning individual components even if not perfect
        """
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            score = 0
            
            # Check each format element
            score += 0.5 if response.count(self.reasoning_start) == 1 else -0.5
            score += 0.5 if response.count(self.reasoning_end) == 1 else -0.5
            score += 0.5 if response.count(self.solution_start) == 1 else -0.5
            score += 0.5 if response.count(self.solution_end) == 1 else -0.5
            
            scores.append(score)
        return scores
    
    def check_answer_correctness(self, prompts, completions, answer, **kwargs):
        """
        Graduated scoring for mathematical accuracy:
        - 3.0: Exact match
        - 1.5: Within 10% (close answer)
        - 0.5: Within 20% (reasonable attempt)
        - -0.5: Wrong answer (penalty for incorrect math)
        """
        responses = [completion[0]["content"] for completion in completions]
        
        # Extract answers from formatted responses
        extracted_responses = [
            guess.group(1) if (guess := self.match_format.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:  # No answer extracted
                scores.append(0)
                continue
                
            # Check exact match first
            if guess.strip() == true_answer.strip():
                scores.append(3.0)
            else:
                # Try numerical comparison
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:      # Within 10%
                        scores.append(1.5)
                    elif 0.8 <= ratio <= 1.2:    # Within 20%
                        scores.append(0.5)
                    else:                         # Wrong answer
                        scores.append(-0.5)
                except (ValueError, ZeroDivisionError):
                    scores.append(-0.5)           # Invalid number
        
        return scores
    
    def check_numbers_extraction(self, prompts, completions, answer, **kwargs):
        """
        Tests the model's ability to extract numerical values from solution sections
        Complementary to exact format matching - focuses on parsing capability
        """
        responses = [completion[0]["content"] for completion in completions]
        
        # Extract numbers from solution sections
        extracted_responses = [
            guess.group(1) if (guess := self.match_numbers.search(r)) is not None else None
            for r in responses
        ]
        
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:  # No number extracted
                scores.append(0)
                continue
                
            try:
                # Compare numerical values
                true_val = float(true_answer.strip())
                guess_val = float(guess.strip())
                
                scores.append(1.5 if guess_val == true_val else 0.0)
            except (ValueError, TypeError):
                scores.append(0)  # Invalid number format
        
        return scores


class DatasetProcessor:
    """Handles GSM8K dataset loading and processing"""
    
    def __init__(self, reward_functions: RewardFunctions):
        self.reward_functions = reward_functions
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for mathematical reasoning"""
        return f"""You are a mathematical reasoning assistant.
When given a math problem:
1. Show your step-by-step work between {self.reward_functions.reasoning_start} and {self.reward_functions.reasoning_end}
2. Provide your final numerical answer between {self.reward_functions.solution_start} and {self.reward_functions.solution_end}
3. Be precise and show all calculation steps clearly."""
    
    def extract_hash_answer(self, text: str) -> Optional[str]:
        """Extract numerical answer from GSM8K format (#### marker)"""
        if "####" not in text:
            return None
        return text.split("####")[1].strip()
    
    def process_dataset_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Convert GSM8K example to conversation format for GRPO training"""
        question = example["question"]
        answer = self.extract_hash_answer(example["answer"])
        
        # Create conversation format
        prompt = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        
        return {
            "prompt": prompt,
            "answer": answer,
        }
    
    def load_and_process_dataset(self, split: str = "train", max_examples: Optional[int] = None):
        """Load and process the GSM8K dataset"""
        logger.info("Loading GSM8K mathematical reasoning dataset...")
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        
        # Limit dataset size if specified
        if max_examples:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        
        # Process dataset
        dataset = dataset.map(self.process_dataset_example)
        
        logger.info(f"Dataset loaded and processed!")
        logger.info(f"Training examples: {len(dataset):,}")
        logger.info(f"Sample question: {dataset[0]['prompt'][1]['content'][:100]}...")
        logger.info(f"Sample answer: {dataset[0]['answer']}")
        
        return dataset


class ModelSetup:
    """Handles model loading, quantization, and LoRA configuration"""
    
    def __init__(self, config: GRPOTrainingConfig):
        self.config = config
    
    def create_quantization_config(self) -> BitsAndBytesConfig:
        """Create 4-bit quantization configuration"""
        # Only use quantization if CUDA is available
        if not torch.cuda.is_available():
            logger.info("⚠️  CUDA not available, skipping quantization")
            return None
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        logger.info("✅ 4-bit quantization configured")
        logger.info("   Memory reduction: ~75% vs FP16")
        
        return bnb_config
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Create quantization config
        bnb_config = self.create_quantization_config()
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16
        else:
            # CPU configuration
            model_kwargs["torch_dtype"] = torch.float32
            
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        return model, tokenizer
    
    def apply_lora(self, model):
        """Apply LoRA (Low-Rank Adaptation) to the model"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info("🔧 Applying LoRA adaptation to model...")
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        logger.info("📊 LoRA Training Parameters Summary:")
        model.print_trainable_parameters()
        
        return model


class GRPOTrainingPipeline:
    """Main GRPO training pipeline"""
    
    def __init__(self, config: GRPOTrainingConfig):
        self.config = config
        self.reward_functions = RewardFunctions()
        self.dataset_processor = DatasetProcessor(self.reward_functions)
        self.model_setup = ModelSetup(config)
        
    def check_gpu_environment(self):
        """Check and display GPU environment information"""
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            logger.info(f"Current GPU: {torch.cuda.current_device()}")
            logger.info(f"GPU name: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("⚠️  No GPU available. This training requires a GPU for efficient execution.")
            logger.warning("Consider using a GPU-enabled environment for optimal performance.")
    
    def setup_experiment_tracking(self) -> str:
        """Setup trackio experiment tracking"""
        if not self.config.use_trackio:
            return ""
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"qwen2.5-3b-gsm8k-grpo-{timestamp}"
        
        trackio.init(
            project=self.config.project_name,
            name=run_name,
            config={
                # Model configuration
                "model_name": self.config.model_name,
                "dataset": "GSM8K",
                "technique": "GRPO + LoRA + 4-bit",
                
                # Training configuration
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.per_device_train_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "effective_batch_size": self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps,
                "max_steps": self.config.max_steps,
                
                # LoRA configuration
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                
                # Generation configuration
                "num_generations": self.config.num_generations,
                "max_prompt_length": self.config.max_prompt_length,
                "max_completion_length": self.config.max_completion_length,
                
                # Reward configuration
                "num_reward_functions": 4,
            }
        )
        
        logger.info("🎯 GRPO Configuration Summary:")
        logger.info(f"   Learning rate: {self.config.learning_rate}")
        logger.info(f"   Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"   Training steps: {self.config.max_steps}")
        logger.info(f"   Generations per step: {self.config.num_generations}")
        logger.info("✅ Trackio experiment tracking initialized")
        logger.info(f"📊 Run name: {run_name}")
        
        return run_name
    
    def create_training_config(self) -> GRPOConfig:
        """Create GRPO training configuration"""
        training_args = GRPOConfig(
            # Learning configuration
            learning_rate=self.config.learning_rate,
            
            # Batch configuration
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Sequence length configuration
            max_prompt_length=self.config.max_prompt_length,
            max_completion_length=self.config.max_completion_length,
            
            # Training steps configuration
            max_steps=self.config.max_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            
            # Output configuration
            output_dir=self.config.output_dir,
            max_grad_norm=self.config.max_grad_norm,
            report_to="trackio" if self.config.use_trackio else None,
            
            # Generation configuration
            num_generations=self.config.num_generations,
            
            # CPU compatibility
            bf16=False if not torch.cuda.is_available() else None,
            fp16=False if not torch.cuda.is_available() else None,
        )
        
        return training_args
    
    def train(self, max_examples: Optional[int] = None):
        """Execute the complete GRPO training pipeline"""
        logger.info("🚀 Starting GRPO Mathematical Reasoning Training Pipeline")
        
        # Check GPU environment
        self.check_gpu_environment()
        
        # Setup experiment tracking
        run_name = self.setup_experiment_tracking()
        
        # Load and process dataset
        dataset = self.dataset_processor.load_and_process_dataset(max_examples=max_examples)
        
        # Load model and tokenizer
        model, tokenizer = self.model_setup.load_model_and_tokenizer()
        
        # Apply LoRA
        model = self.model_setup.apply_lora(model)
        
        # Create training configuration
        training_args = self.create_training_config()
        
        # Initialize trainer
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[
                self.reward_functions.match_format_exactly,
                self.reward_functions.match_format_approximately,
                self.reward_functions.check_answer_correctness,
                self.reward_functions.check_numbers_extraction,
            ],
            args=training_args,
            train_dataset=dataset,
        )
        
        logger.info("✅ GRPO Trainer initialized successfully!")
        logger.info(f"📊 Training dataset: {len(dataset):,} examples")
        logger.info(f"🎯 Reward functions: {len(trainer.reward_funcs)} active")
        logger.info(f"📈 Trackio integration: {'Enabled' if self.config.use_trackio else 'Disabled'}")
        logger.info(f"🔄 Ready for training with {training_args.num_generations} generations per step")
        
        # Start training
        logger.info("🚀 Starting GRPO training...")
        logger.info("📊 Monitor metrics: reward scores, KL divergence, policy gradients")
        if self.config.use_trackio:
            logger.info("🔍 Trackio will log: losses, rewards, learning rate, gradients")
        
        trainer.train()
        
        # Finish experiment tracking
        if self.config.use_trackio:
            trackio.finish()
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"💾 Model saved to: {training_args.output_dir}")
        
        return model, tokenizer, trainer
    
    def cleanup_resources(self):
        """Clean up GPU memory and resources"""
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("✅ GPU memory cache cleared")
        logger.info("✅ Python garbage collection completed")
        logger.info("🧹 Resources freed for other processes")


class ModelEvaluator:
    """Handles model evaluation and testing"""
    
    def __init__(self, model, tokenizer, reward_functions: RewardFunctions):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_functions = reward_functions
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for evaluation"""
        return f"""You are a mathematical reasoning assistant.
When given a math problem:
1. Show your step-by-step work between {self.reward_functions.reasoning_start} and {self.reward_functions.reasoning_end}
2. Provide your final numerical answer between {self.reward_functions.solution_start} and {self.reward_functions.solution_end}
3. Be precise and show all calculation steps clearly."""
    
    def test_model(self, question: str, max_length: int = 512) -> str:
        """
        Test the trained model on mathematical questions
        
        Args:
            question (str): Mathematical problem to solve
            max_length (int): Maximum tokens to generate
            
        Returns:
            str: Model's structured response with reasoning and solution
        """
        # Create conversation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        logger.info(f"🤔 Processing: {question}")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=True,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(text):].strip()
        
        return generated_text
    
    def evaluate_response(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """Evaluate a model response against expected answer"""
        # Check format compliance
        has_reasoning = (self.reward_functions.reasoning_start in response and 
                        self.reward_functions.reasoning_end in response)
        has_solution = (self.reward_functions.solution_start in response and 
                       self.reward_functions.solution_end in response)
        
        evaluation = {
            "has_reasoning_section": has_reasoning,
            "has_solution_section": has_solution,
            "format_compliant": has_reasoning and has_solution,
            "extracted_answer": None,
            "is_correct": False,
            "error_message": None
        }
        
        # Extract and evaluate answer
        if has_solution:
            try:
                solution_text = response.split(self.reward_functions.solution_start)[1].split(self.reward_functions.solution_end)[0].strip()
                evaluation["extracted_answer"] = solution_text
                
                # Extract numbers for comparison
                extracted_number = ''.join(filter(str.isdigit, solution_text))
                expected_number = ''.join(filter(str.isdigit, expected_answer))
                evaluation["is_correct"] = extracted_number == expected_number
                
            except Exception as e:
                evaluation["error_message"] = f"Could not extract solution: {str(e)}"
        
        return evaluation
    
    def run_evaluation_suite(self, test_questions: List[Dict[str, str]]) -> Dict[str, Any]:
        """Run evaluation on a suite of test questions"""
        results = []
        
        for i, test_case in enumerate(test_questions):
            question = test_case["question"]
            expected_answer = test_case["answer"]
            
            logger.info(f"Evaluating question {i+1}/{len(test_questions)}")
            
            # Generate response
            response = self.test_model(question)
            
            # Evaluate response
            evaluation = self.evaluate_response(response, expected_answer)
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "model_response": response,
                "evaluation": evaluation
            })
        
        # Calculate summary statistics
        total_questions = len(results)
        format_compliant = sum(1 for r in results if r["evaluation"]["format_compliant"])
        correct_answers = sum(1 for r in results if r["evaluation"]["is_correct"])
        
        summary = {
            "total_questions": total_questions,
            "format_compliance_rate": format_compliant / total_questions,
            "accuracy_rate": correct_answers / total_questions,
            "results": results
        }
        
        logger.info(f"📊 Evaluation Summary:")
        logger.info(f"   Total questions: {total_questions}")
        logger.info(f"   Format compliance: {format_compliant}/{total_questions} ({summary['format_compliance_rate']:.2%})")
        logger.info(f"   Accuracy: {correct_answers}/{total_questions} ({summary['accuracy_rate']:.2%})")
        
        return summary


def main():
    """Main function to run the GRPO training pipeline"""
    # Create configuration
    config = GRPOTrainingConfig()
    
    # You can modify these parameters as needed
    config.max_steps = 50  # Adjust based on your needs
    config.use_trackio = True  # Set to False if you don't want experiment tracking
    
    # Create and run training pipeline
    pipeline = GRPOTrainingPipeline(config)
    
    try:
        # Train the model (limit to 1000 examples for faster training)
        model, tokenizer, trainer = pipeline.train(max_examples=1000)
        
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
        
        # Display sample result
        if evaluation_results["results"]:
            sample_result = evaluation_results["results"][0]
            logger.info(f"\n📝 Sample Evaluation:")
            logger.info(f"Question: {sample_result['question']}")
            logger.info(f"Expected: {sample_result['expected_answer']}")
            logger.info(f"Model Response:\n{sample_result['model_response']}")
            logger.info(f"Evaluation: {sample_result['evaluation']}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Clean up resources
        pipeline.cleanup_resources()


if __name__ == "__main__":
    main()