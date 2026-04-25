import os
import random
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from question_generator import QuestionGenerator
from student_model import StudentProfile
from expert_simulator import ExpertSimulator
from reward import RewardManager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Shared AI Model ---
AI_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINED_MODEL_PATH = "./adaptive_tutor_trained"

ai_model = None
ai_tokenizer = None
AI_LOADED = False

def load_ai_model():
    global ai_model, ai_tokenizer, AI_LOADED
    if AI_LOADED: return True
    
    target = TRAINED_MODEL_PATH if os.path.exists(TRAINED_MODEL_PATH) else AI_MODEL_NAME
    try:
        print(f"Loading AI model: {target}...")
        ai_tokenizer = AutoTokenizer.from_pretrained(target)
        
        # Quantization for GPU, otherwise standard load
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            ai_model = AutoModelForCausalLM.from_pretrained(target, quantization_config=bnb_config, device_map="auto")
        else:
            ai_model = AutoModelForCausalLM.from_pretrained(target, device_map="cpu", torch_dtype=torch.float32)
        
        AI_LOADED = True
        return True
    except Exception as e:
        print(f"Failed to load AI model: {e}")
        return False

# --- Shared Models ---

class TutorAction(BaseModel):
    action_type: str = Field(..., description="ask_question | give_hint | explain_concept | increase_difficulty | decrease_difficulty")
    content: str = Field(default="", description="Question/hint/explanation text")
    target_concept: str = Field(default="", description="Target concept")
    difficulty: int = Field(default=1, ge=1, le=5, description="Difficulty 1-5")

class TutorObservation(BaseModel):
    student_profile: Dict[str, Any] = Field(default_factory=dict)
    current_topic: str = Field(default="")
    current_difficulty: int = Field(default=1)
    student_correct: bool = Field(default=False)
    consecutive_correct: int = Field(default=0)
    consecutive_wrong: int = Field(default=0)
    expert_feedback: Optional[str] = Field(default=None)
    expert_preference_changed: bool = Field(default=False)
    session_progress: float = Field(default=0.0)
    mastery_achieved: bool = Field(default=False)
    reward: float = Field(default=0.0)
    done: bool = Field(default=False)
    info: Dict[str, Any] = Field(default_factory=dict)

try:
    from openenv.core.env_server import Environment, create_fastapi_app
except (ImportError, ModuleNotFoundError):
    # Minimal fallback if openenv-core is not found
    class Environment:
        def __init__(self, *args, **kwargs): pass
    def create_fastapi_app(factory, action_model, obs_model):
        from fastapi import FastAPI
        return FastAPI()

# --- Shared Environment ---

class AdaptiveTutorEnv(Environment):
    MAX_STEPS = 20
    MASTERY_THRESHOLD = 0.8

    def __init__(self):
        super().__init__()
        self.question_gen = QuestionGenerator(subjects_dir="subjects")
        self.expert_sim = ExpertSimulator()
        self.student = None
        self.current_subject = ""
        self.current_difficulty = 1
        self.step_count = 0
        self.last_student_correct = False
        self.last_expert_feedback = None
        self.last_expert_pref_changed = False
        self.last_reward = 0.0
        self.last_reward_breakdown = {}
        self.mastery_concepts_rewarded = set()
        self.episode_done = False

    def reset(self) -> TutorObservation:
        self.current_subject = self.question_gen.get_random_subject()
        concepts = self.question_gen.get_concepts_for_subject(self.current_subject)
        self.student = StudentProfile(name="Simulated Student")
        self.student.initialize_knowledge(concepts)
        self.current_difficulty = 1
        self.step_count = 0
        self.last_student_correct = False
        self.last_expert_feedback = None
        self.last_expert_pref_changed = False
        self.last_reward = 0.0
        self.last_reward_breakdown = {}
        self.mastery_concepts_rewarded = set()
        self.episode_done = False
        self.question_gen.reset_asked_questions()
        self.expert_sim.reset()
        return self._make_observation()

    def step(self, action: TutorAction) -> TutorObservation:
        if self.episode_done or self.student is None:
            return self._make_observation()
        self.step_count += 1
        
        # Sync difficulty
        if action.action_type == "increase_difficulty":
            self.current_difficulty = min(5, self.current_difficulty + 1)
        elif action.action_type == "decrease_difficulty":
            self.current_difficulty = max(1, self.current_difficulty - 1)
        else:
            self.current_difficulty = action.difficulty

        # Simulate
        target_concept = action.target_concept
        student_correct = False
        if action.action_type in ("ask_question", "increase_difficulty", "decrease_difficulty"):
            question = self.question_gen.get_question(self.current_subject, self.current_difficulty, target_concept)
            if question:
                target_concept = question.get("concept", target_concept)
                student_correct = self.student.simulate_response(target_concept, self.current_difficulty)
                self.student.update_knowledge(target_concept, student_correct)
        
        self.last_student_correct = student_correct
        expert_feedback, pref_changed = self.expert_sim.step(self.current_subject, action.model_dump())
        self.last_expert_feedback = expert_feedback
        self.last_expert_pref_changed = pref_changed

        # Reward
        concept_mastery = self.student.knowledge_map.get(target_concept, 0.3)
        overall_mastery = self.student.get_overall_mastery()
        total_reward, breakdown = RewardManager.calculate_total_reward(
            student_correct=student_correct,
            concept=target_concept,
            knowledge_map=self.student.knowledge_map,
            chosen_difficulty=self.current_difficulty,
            student_mastery=concept_mastery,
            consecutive_correct=self.student.consecutive_correct,
            consecutive_wrong=self.student.consecutive_wrong,
            expert_feedback=expert_feedback,
            tutor_adapted=random.random() < 0.4,
            overall_mastery=overall_mastery,
            steps_taken=self.step_count,
            max_steps=self.MAX_STEPS
        )
        self.last_reward = total_reward
        self.last_reward_breakdown = breakdown

        if overall_mastery >= self.MASTERY_THRESHOLD or self.step_count >= self.MAX_STEPS:
            self.episode_done = True
        return self._make_observation()

    def _make_observation(self) -> TutorObservation:
        profile = self.student.get_profile_summary() if self.student else {}
        overall_mastery = self.student.get_overall_mastery() if self.student else 0.0
        return TutorObservation(
            student_profile=profile,
            current_topic=self.current_subject,
            current_difficulty=self.current_difficulty,
            student_correct=self.last_student_correct,
            consecutive_correct=self.student.consecutive_correct if self.student else 0,
            consecutive_wrong=self.student.consecutive_wrong if self.student else 0,
            expert_feedback=self.last_expert_feedback,
            expert_preference_changed=self.last_expert_pref_changed,
            session_progress=self.step_count / self.MAX_STEPS,
            mastery_achieved=overall_mastery >= self.MASTERY_THRESHOLD,
            reward=self.last_reward,
            done=self.episode_done,
            info=self.last_reward_breakdown
        )

# Global Instance
env_instance = AdaptiveTutorEnv()
