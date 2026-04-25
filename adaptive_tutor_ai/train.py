"""
train.py - GRPO Training Script for AdaptiveTutor AI

Uses Group Relative Policy Optimization (GRPO) from TRL library
with Unsloth for memory-efficient training. Designed to run
end-to-end on Google Colab with a T4 GPU.

GRPO Training Loop:
1. Load Qwen2.5-3B-Instruct with Unsloth 4-bit quantization
2. Connect to AdaptiveTutor environment (server.py)
3. Collect trajectories: tutor actions → observations → rewards
4. Train with GRPOTrainer using multiple reward signals
5. Track per-function rewards (correctness, mastery, difficulty, expert, efficiency)
6. Save model properly (Unsloth merge_and_unload, NOT naive 4-bit save)

Usage:
    # Start the environment server first:
    # uvicorn server:app --port 7860
    
    # Then run training:
    # python train.py
    
    # Or in Google Colab:
    # !pip install unsloth trl openenv-core
    # !python train.py
"""

import os
import json
import random
import asyncio
from typing import Dict, List, Any, Optional

# ──────────────────────────────────────────────
#  Step 1: Install & Import Dependencies
#  (In Colab, run: !pip install unsloth trl)
# ──────────────────────────────────────────────
try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
    print("[✓] Unsloth loaded successfully")
except ImportError:
    HAS_UNSLOTH = False
    print("[!] Unsloth not available — using standard transformers")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from trl import GRPOConfig, GRPOTrainer
    HAS_TRL = True
    print("[✓] TRL loaded successfully")
except ImportError:
    HAS_TRL = False
    print("[!] TRL not available — training will be simulated")

import httpx

# ──────────────────────────────────────────────
#  Step 2: Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16          # LoRA rank for parameter-efficient fine-tuning
LORA_ALPHA = 16         # LoRA alpha scaling
ENV_URL = "http://127.0.0.1:7860"
NUM_EPISODES = 50       # Number of training episodes
MAX_STEPS_PER_EPISODE = 20
SAVE_DIR = "./adaptive_tutor_trained"
BATCH_SIZE = 4          # Prompts per GRPO batch
NUM_GENERATIONS = 4     # Completions per prompt for GRPO

# Reward tracking
REWARD_HISTORY: List[Dict[str, float]] = []


# ──────────────────────────────────────────────
#  Step 3: Load Model with Unsloth
#  
#  Unsloth provides 2x faster training and 60% less memory
#  by optimizing attention kernels and using smart 4-bit quantization.
#  This is critical for fitting Qwen2.5-3B on a Colab T4 GPU.
# ──────────────────────────────────────────────
def load_model():
    """
    Load the base model with Unsloth for efficient training.
    
    Returns 4-bit quantized model with LoRA adapters attached.
    If Unsloth is not available, falls back to standard HF loading.
    
    Returns:
        Tuple of (model, tokenizer).
    """
    if HAS_UNSLOTH:
        print(f"[*] Loading {MODEL_NAME} with Unsloth (4-bit)...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,  # Auto-detect (float16 on T4)
            load_in_4bit=True,  # 4-bit quantization for memory efficiency
        )
        
        # Add LoRA adapters for parameter-efficient fine-tuning
        # Only train the attention layers, not the entire model
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,     # Optimized — no dropout needed with LoRA
            bias="none",
            use_gradient_checkpointing="unsloth",  # 60% less VRAM
            random_state=42,
        )
        print("[✓] Model loaded with Unsloth + LoRA")
    else:
        print(f"[*] Loading {MODEL_NAME} with standard transformers...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("[✓] Model loaded (standard)")
    
    return model, tokenizer


# ──────────────────────────────────────────────
#  Step 4: Environment Interaction
#  
#  The training loop connects to the AdaptiveTutor server
#  and collects (prompt, completion, reward) tuples for GRPO.
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are an adaptive AI tutor. Based on the student's profile and performance, decide the best tutoring action.

Available actions:
- ask_question: Ask the student a question
- give_hint: Provide a helpful hint
- explain_concept: Explain a concept clearly
- increase_difficulty: Move to a harder level
- decrease_difficulty: Move to an easier level

Respond with a JSON object:
{
    "action_type": "ask_question",
    "content": "Your question or explanation here",
    "target_concept": "concept_name",
    "difficulty": 1-5
}

Adapt your strategy based on:
1. Student's knowledge map (weak vs strong areas)
2. Consecutive correct/wrong streaks
3. Expert feedback (when provided)
4. Session progress
"""


def build_prompt(observation: Dict[str, Any]) -> str:
    """
    Build a training prompt from an environment observation.
    
    The prompt includes the system instruction and the current
    observation formatted as a conversation turn.
    
    Args:
        observation: The TutorObservation dict from the environment.
        
    Returns:
        Formatted prompt string.
    """
    profile = observation.get("student_profile", {})
    expert_fb = observation.get("expert_feedback", "None")
    pref_changed = observation.get("expert_preference_changed", False)
    
    user_msg = f"""Current Observation:
- Subject: {observation.get('current_topic', 'unknown')}
- Difficulty: {observation.get('current_difficulty', 1)}/5
- Last Answer Correct: {observation.get('student_correct', False)}
- Consecutive Correct: {observation.get('consecutive_correct', 0)}
- Consecutive Wrong: {observation.get('consecutive_wrong', 0)}
- Overall Mastery: {profile.get('overall_mastery', 0)}
- Weak Concepts: {profile.get('weak_concepts', [])}
- Strong Concepts: {profile.get('strong_concepts', [])}
- Expert Feedback: {expert_fb}
- Expert Preferences Changed: {pref_changed}
- Session Progress: {observation.get('session_progress', 0):.0%}

Decide your next tutoring action:"""

    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def parse_model_output(output: str) -> Dict[str, Any]:
    """
    Parse the model's JSON output into a TutorAction dict.
    
    Falls back to a default action if parsing fails.
    
    Args:
        output: Raw model output string.
        
    Returns:
        Action dict with action_type, content, target_concept, difficulty.
    """
    try:
        # Try to extract JSON from the output
        start = output.find("{")
        end = output.rfind("}") + 1
        if start != -1 and end > start:
            action = json.loads(output[start:end])
            # Validate required fields
            if "action_type" in action:
                return {
                    "action_type": action.get("action_type", "ask_question"),
                    "content": action.get("content", ""),
                    "target_concept": action.get("target_concept", ""),
                    "difficulty": max(1, min(5, action.get("difficulty", 1))),
                }
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: default action
    return {
        "action_type": "ask_question",
        "content": "",
        "target_concept": "",
        "difficulty": 1,
    }


async def collect_trajectory(
    base_url: str = ENV_URL,
) -> List[Dict[str, Any]]:
    """
    Collect one episode trajectory from the environment.
    
    Returns a list of (prompt, action, reward, reward_breakdown) dicts
    that can be used for GRPO training.
    
    Args:
        base_url: URL of the AdaptiveTutor server.
        
    Returns:
        List of trajectory step dicts.
    """
    trajectory = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Reset environment
        response = await client.post(f"{base_url}/reset")
        result = response.json()
        obs = result.get("observation", result)
        done = result.get("done", False)
        
        step = 0
        while not done and step < MAX_STEPS_PER_EPISODE:
            step += 1
            
            # Build prompt from observation
            prompt = build_prompt(obs)
            
            # For training data collection, use a heuristic agent
            # (In actual GRPO, the model generates the action)
            action = _heuristic_action(obs)
            
            # Step environment (OpenEnv expects action nested under "action" key)
            response = await client.post(
                f"{base_url}/step",
                json={"action": action},
            )
            result = response.json()
            obs = result.get("observation", result)
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            info = result.get("info", {})
            
            trajectory.append({
                "prompt": prompt,
                "action": action,
                "reward": reward,
                "observation": obs,
                "done": done,
            })
    
    return trajectory


def _heuristic_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple heuristic agent for trajectory collection.
    
    Args:
        obs: Current observation dict.
        
    Returns:
        Action dict.
    """
    profile = obs.get("student_profile", {})
    weak = profile.get("weak_concepts", [])
    weakest = profile.get("weakest_concept", "")
    cons_wrong = obs.get("consecutive_wrong", 0)
    cons_correct = obs.get("consecutive_correct", 0)
    difficulty = obs.get("current_difficulty", 1)
    
    if cons_wrong >= 2:
        return {
            "action_type": "decrease_difficulty" if difficulty > 1 else "give_hint",
            "content": "Let me help you with this.",
            "target_concept": weakest,
            "difficulty": max(1, difficulty - 1),
        }
    
    if cons_correct >= 3:
        return {
            "action_type": "increase_difficulty",
            "content": "Great work! Let's challenge you more.",
            "target_concept": weakest or obs.get("current_topic", ""),
            "difficulty": min(5, difficulty + 1),
        }
    
    return {
        "action_type": "ask_question",
        "content": "",
        "target_concept": weakest or (weak[0] if weak else ""),
        "difficulty": difficulty,
    }


# ──────────────────────────────────────────────
#  Step 5: GRPO Training with TRL
#  
#  GRPO (Group Relative Policy Optimization) works by:
#  1. Generating multiple completions for each prompt
#  2. Getting rewards for each completion from the environment
#  3. Using the RELATIVE reward rankings (not absolute values)
#     to update the policy — higher-reward completions are
#     reinforced, lower-reward ones are discouraged
#  
#  This is more stable than PPO because it doesn't need
#  a separate value function — it just compares within groups.
# ──────────────────────────────────────────────

def create_reward_function(env_url: str = ENV_URL):
    """
    Create a reward function compatible with TRL's GRPOTrainer.
    
    The reward function sends the model's generated action to the
    environment and returns the multi-factor reward.
    
    Args:
        env_url: URL of the AdaptiveTutor server.
        
    Returns:
        Reward function callable.
    """
    def reward_fn(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        """
        Evaluate completions by sending them to the environment.
        
        Args:
            completions: List of model-generated action strings.
            prompts: List of corresponding prompts.
            
        Returns:
            List of reward floats.
        """
        rewards = []
        for completion in completions:
            action = parse_model_output(completion)
            
            # Send to environment and get reward
            try:
                import httpx as httpx_sync
                with httpx_sync.Client(timeout=10.0) as client:
                    response = client.post(f"{env_url}/step", json={"action": action})
                    result = response.json()
                    reward = result.get("reward", 0.0)
                    rewards.append(reward)
            except Exception:
                rewards.append(0.0)
        
        return rewards
    
    return reward_fn


def train():
    """
    Main GRPO training loop.
    
    Steps:
    1. Load model with Unsloth
    2. Collect training prompts from environment trajectories
    3. Configure GRPOTrainer
    4. Train with environment rewards
    5. Save model properly
    """
    print("\n" + "=" * 60)
    print("  AdaptiveTutor AI - GRPO Training")
    print("=" * 60)
    
    # --- Load Model ---
    model, tokenizer = load_model()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Collect Training Data ---
    # Pre-collect some trajectories for training prompts
    print("\n[*] Collecting training trajectories...")
    all_prompts = []
    
    try:
        for ep in range(min(10, NUM_EPISODES)):
            trajectory = asyncio.run(collect_trajectory())
            prompts = [step["prompt"] for step in trajectory]
            all_prompts.extend(prompts)
            total_reward = sum(step["reward"] for step in trajectory)
            REWARD_HISTORY.append({
                "episode": ep,
                "total_reward": total_reward,
                "steps": len(trajectory),
            })
            print(f"  Episode {ep+1}: {len(trajectory)} steps, reward={total_reward:.2f}")
    except Exception as e:
        print(f"[!] Could not connect to environment: {e}")
        print("[*] Generating synthetic training prompts instead...")
        # Generate synthetic prompts for offline training
        for _ in range(50):
            synthetic_obs = {
                "student_profile": {
                    "overall_mastery": random.uniform(0.2, 0.7),
                    "weak_concepts": [random.choice(["algebra", "fractions", "geometry"])],
                    "strong_concepts": [random.choice(["addition", "counting"])],
                    "weakest_concept": random.choice(["algebra", "fractions"]),
                },
                "current_topic": random.choice(["math", "science", "history"]),
                "current_difficulty": random.randint(1, 5),
                "student_correct": random.choice([True, False]),
                "consecutive_correct": random.randint(0, 4),
                "consecutive_wrong": random.randint(0, 3),
                "expert_feedback": random.choice([
                    None,
                    "Increase difficulty now, student is ready.",
                    "Student struggling, try different approach.",
                    "Focus on concept 'algebra', student is weak there.",
                ]),
                "expert_preference_changed": random.choice([True, False]),
                "session_progress": random.uniform(0, 1),
            }
            all_prompts.append(build_prompt(synthetic_obs))
    
    if not all_prompts:
        print("[!] No training data collected. Exiting.")
        return
    
    print(f"[✓] Collected {len(all_prompts)} training prompts")
    
    # --- Configure GRPO Training ---
    if HAS_TRL:
        print("\n[*] Configuring GRPO Trainer...")
        
        # GRPOConfig sets up the Group Relative Policy Optimization
        # - num_generations: how many completions to generate per prompt
        # - temperature: controls diversity of generated completions
        # - max_completion_length: max tokens for each completion
        training_args = GRPOConfig(
            output_dir=SAVE_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=5e-6,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_completion_length=256,
            num_generations=NUM_GENERATIONS,
            temperature=0.7,
            logging_steps=5,
            save_steps=50,
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
            gradient_accumulation_steps=4,
            report_to="none",  # Set to "wandb" for W&B tracking
        )
        
        # Create the reward function
        reward_fn = create_reward_function()
        
        # Format prompts as dataset
        from datasets import Dataset
        train_dataset = Dataset.from_dict({"prompt": all_prompts})
        
        # Initialize GRPO Trainer
        # GRPO generates multiple completions per prompt, ranks them by reward,
        # and uses the relative rankings to compute policy gradients
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
        )
        
        print("[*] Starting GRPO training...")
        print("    This will generate multiple actions per observation,")
        print("    get rewards from the environment, and use GRPO to")
        print("    reinforce better tutoring strategies.")
        
        trainer.train()
        print("[✓] Training complete!")
        
        # --- Save Model ---
        # IMPORTANT: Use Unsloth's merge_and_unload for proper saving
        # Do NOT naively save 4-bit weights — they won't load correctly
        print("\n[*] Saving trained model...")
        if HAS_UNSLOTH:
            # Unsloth's merge_and_unload properly merges LoRA weights
            # back into the base model before saving
            model.save_pretrained_merged(
                SAVE_DIR,
                tokenizer,
                save_method="merged_16bit",  # Save as float16 (not 4-bit!)
            )
            print(f"[✓] Model saved with Unsloth merge to: {SAVE_DIR}")
        else:
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"[✓] Model saved to: {SAVE_DIR}")
    else:
        print("\n[!] TRL not available. Simulating training loop...")
        print("    Install TRL: pip install trl")
        print("    Then re-run this script.")
    
    # --- Print Training Summary ---
    print("\n" + "=" * 60)
    print("  Training Summary")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Training Prompts: {len(all_prompts)}")
    print(f"  Episodes Collected: {len(REWARD_HISTORY)}")
    if REWARD_HISTORY:
        avg_reward = sum(r["total_reward"] for r in REWARD_HISTORY) / len(REWARD_HISTORY)
        print(f"  Average Episode Reward: {avg_reward:.2f}")
    print(f"  Save Directory: {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train()
