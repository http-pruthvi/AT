import os
import sys
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

# Import our environment client
from client import AdaptiveTutorEnv, TutorAction

BASE_URL = "https://http-pruthvi-adaptive-tutor-ai.hf.space"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load model with 4-bit quantization for Colab T4
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# OpenEnv reward function - connects to HF Space
def openenv_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        try:
            with AdaptiveTutorEnv(base_url=BASE_URL).sync() as env:
                obs = env.reset()
                action = TutorAction(
                    question=completion,
                    difficulty=obs.recommended_difficulty,
                    concept=obs.weak_concepts[0] if obs.weak_concepts else "general",
                    explanation=""
                )
                result = env.step(action)
                rewards.append(float(result.reward))
        except Exception as e:
            print(f"Reward error: {e}")
            rewards.append(0.0)
    return rewards

# Training dataset - prompts for tutor to generate questions
def make_dataset():
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

dataset = make_dataset()

# GRPO Config
config = GRPOConfig(
    output_dir="./grpo_adaptive_tutor",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_completion_length=256,
    num_generations=4,
    logging_steps=5,
    save_steps=50,
    warmup_steps=10,
    report_to="none",
)

# Train
trainer = GRPOTrainer(
    model=model,
    reward_funcs=openenv_reward,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

print("Starting GRPO training on AdaptiveTutor environment...")
print(f"Environment: {BASE_URL}")
trainer.train()

# Save and push to Hub
model.save_pretrained("./adaptive_tutor_trained")
tokenizer.save_pretrained("./adaptive_tutor_trained")
print("Training complete! Model saved.")
