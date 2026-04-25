import os
import sys
import httpx

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from shared import TutorAction

BASE_URL = os.getenv("ADAPTIVE_TUTOR_ENV_URL", "https://http-pruthvi-adaptive-tutor-ai.hf.space")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def _step_env_once(completion: str) -> float:
    """Get reward by sending one action to the OpenEnv HTTP server."""
    try:
        with httpx.Client(timeout=20.0) as client:
            reset_res = client.post(f"{BASE_URL}/reset")
            reset_res.raise_for_status()
            reset_payload = reset_res.json()
            obs = reset_payload.get("observation", reset_payload)
            current_difficulty = int(obs.get("current_difficulty", 1))
            profile = obs.get("student_profile", {}) or {}
            target_concept = profile.get("weakest_concept") or obs.get("current_topic", "general")

            action = TutorAction(
                action_type="ask_question",
                content=completion,
                difficulty=max(1, min(5, current_difficulty)),
                target_concept=target_concept,
            )
            step_res = client.post(f"{BASE_URL}/step", json={"action": action.model_dump()})
            step_res.raise_for_status()
            step_payload = step_res.json()
            return float(step_payload.get("reward", 0.0))
    except Exception as e:
        print(f"Reward error: {e}")
        return 0.0

def openenv_reward(completions, **kwargs):
    """OpenEnv reward function for GRPO training."""
    rewards = []
    for completion in completions:
        rewards.append(_step_env_once(completion))
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
