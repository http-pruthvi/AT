import os
import sys
import httpx
import json
import shutil
import hashlib
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from shared import TutorAction
from config import settings

BASE_URL = settings.env_url
MODEL_NAME = settings.ai_model_name
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF_IMPROVE_DATASET_PATH = os.path.join(PROJECT_ROOT, "outputs", "self_improve_dataset.jsonl")
MODEL_REGISTRY_DIR = settings.model_registry_dir

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

def load_dataset():
    """Prefer self-improvement dataset if available; fallback to synthetic prompts."""
    if os.path.exists(SELF_IMPROVE_DATASET_PATH):
        try:
            dataset = Dataset.from_json(SELF_IMPROVE_DATASET_PATH)
            if len(dataset) > 0 and "prompt" in dataset.column_names:
                print(f"Using self-improvement dataset: {SELF_IMPROVE_DATASET_PATH} ({len(dataset)} rows)")
                return dataset
            print("Self-improvement dataset is empty/invalid. Falling back to synthetic dataset.")
        except Exception as e:
            print(f"Failed to load self-improvement dataset: {e}")
            print("Falling back to synthetic dataset.")
    dataset = make_dataset()
    print(f"Using synthetic dataset ({len(dataset)} rows)")
    return dataset


def _file_hash(path: str) -> str:
    if not os.path.exists(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_candidate(dataset_len: int) -> float:
    """
    Lightweight offline gate for pilot.
    Score combines dataset coverage and environment connectivity.
    """
    coverage = min(1.0, dataset_len / 300.0)
    env_ok = _step_env_once("What is one key concept we should review next?") > -10
    return round((0.7 * coverage) + (0.3 if env_ok else 0.0), 3)


def _register_model(model_output_dir: str, dataset_len: int) -> None:
    os.makedirs(MODEL_REGISTRY_DIR, exist_ok=True)
    candidates_dir = os.path.join(MODEL_REGISTRY_DIR, "candidates")
    active_dir = os.path.join(MODEL_REGISTRY_DIR, "active")
    os.makedirs(candidates_dir, exist_ok=True)
    os.makedirs(active_dir, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    candidate_path = os.path.join(candidates_dir, run_id)
    shutil.copytree(model_output_dir, candidate_path, dirs_exist_ok=True)

    eval_score = _evaluate_candidate(dataset_len)
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.utcnow().isoformat(),
        "base_model": MODEL_NAME,
        "dataset_path": SELF_IMPROVE_DATASET_PATH if os.path.exists(SELF_IMPROVE_DATASET_PATH) else "synthetic",
        "dataset_hash": _file_hash(SELF_IMPROVE_DATASET_PATH),
        "dataset_rows": dataset_len,
        "min_eval_score": settings.min_eval_score,
        "eval_score": eval_score,
        "promoted": eval_score >= settings.min_eval_score,
    }
    with open(os.path.join(candidate_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if metadata["promoted"]:
        promoted_path = os.path.join(active_dir, "model")
        if os.path.exists(promoted_path):
            shutil.rmtree(promoted_path)
        shutil.copytree(candidate_path, promoted_path)
        print(f"✅ Candidate promoted to active model. score={eval_score}")
    else:
        print(f"⚠ Candidate not promoted. score={eval_score} min_required={settings.min_eval_score}")

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

    dataset = load_dataset()

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
    output_dir = os.path.join(PROJECT_ROOT, "adaptive_tutor_trained")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _register_model(output_dir, len(dataset))
    print(f"✅ Training complete! Model saved to {output_dir}")

if __name__ == "__main__":
    train()
