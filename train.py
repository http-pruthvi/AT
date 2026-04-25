import os
import sys
import json

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# Import our environment client
# We use the local server if it's running, otherwise fall back to HF Space
try:
    from client import AdaptiveTutorEnv, TutorAction
    LOCAL_SERVER_AVAILABLE = True
except ImportError:
    LOCAL_SERVER_AVAILABLE = False

BASE_URL = "https://http-pruthvi-adaptive-tutor-ai.hf.space"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def openenv_reward(completions, **kwargs):
    """OpenEnv reward function - connects to the environment server."""
    rewards = []
    for completion in completions:
        try:
            # Connect to local or remote environment
            with AdaptiveTutorEnv(base_url=BASE_URL).sync() as env:
                obs = env.reset()
                action = TutorAction(
                    action_type="ask_question",
                    question=completion,
                    difficulty=obs.recommended_difficulty if hasattr(obs, 'recommended_difficulty') else 1,
                    concept=obs.weak_concepts[0] if (hasattr(obs, 'weak_concepts') and obs.weak_concepts) else "general",
                    explanation=""
                )
                result = env.step(action)
                rewards.append(float(result.reward))
        except Exception as e:
            print(f"Reward error: {e}")
            rewards.append(0.0)
    return rewards

def make_dataset():
    """Create a small diverse dataset for training."""
    prompts = []
    subjects = ["Math", "Science", "History"]
    concepts = {
        "Math": ["algebra", "geometry", "calculus"],
        "Science": ["photosynthesis", "cell division", "Newton laws"],
        "History": ["World War 2", "Indian independence", "Renaissance"]
    }
    for subject in subjects:
        for concept in concepts[subject]:
            for difficulty in ["easy", "medium", "hard"]:
                prompt = f"""<|system|>
You are an adaptive AI tutor. Generate a single clear question.
</s>
<|user|>
Subject: {subject}
Concept: {concept}
Difficulty: {difficulty}
Student mastery: 45%
Generate ONE question only. No explanation.
</s>
<|assistant|>"""
                prompts.append({"prompt": prompt})
    return Dataset.from_list(prompts * 3)

def train():
    print(f"Checking for GPU...")
    if not torch.cuda.is_available():
        print("CUDA not available. Training will be extremely slow on CPU.")
        device_map = "cpu"
    else:
        print(f"Found GPU: {torch.cuda.get_device_name(0)}")
        device_map = "auto"

    # 4-bit quantization to save VRAM (Crucial for 4GB GPUs)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )

    dataset = make_dataset()

    # GRPO Config - Optimized for local 4GB-8GB VRAM
    config = GRPOConfig(
        output_dir="./grpo_adaptive_tutor",
        num_train_epochs=1,
        per_device_train_batch_size=1, # Smaller batch
        gradient_accumulation_steps=8, # Compensate for small batch
        learning_rate=1e-5,
        max_completion_length=128,
        num_generations=2, # Fewer generations for speed
        logging_steps=1,
        save_steps=20,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=openenv_reward,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("🚀 Starting GRPO training on AdaptiveTutor...")
    trainer.train()

    # Save
    model.save_pretrained("./adaptive_tutor_trained")
    tokenizer.save_pretrained("./adaptive_tutor_trained")
    print("✅ Training complete! Model saved to ./adaptive_tutor_trained")

if __name__ == "__main__":
    train()
