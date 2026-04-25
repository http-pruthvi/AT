"""
server.py - OpenEnv Environment Server for AdaptiveTutor AI

FastAPI server implementing the OpenEnv protocol (reset/step/state).
Orchestrates the student simulation, question generation, expert
feedback, and multi-factor reward calculation.

Episode flow:
1. reset() → initialize fresh student + load random subject
2. step(action) → evaluate tutor action → update state → return obs + reward
3. Episode ends when student reaches mastery (0.8+) OR 20 steps done
4. Expert feedback arrives randomly every 3-5 steps
"""

import os
import copy
import random
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware
from openenv.core.env_server import Environment, create_fastapi_app

from student_model import StudentProfile
from question_generator import QuestionGenerator
from expert_simulator import ExpertSimulator
from reward import RewardManager


# ──────────────────────────────────────────────
#  Pydantic Models (duplicated here for OpenEnv compatibility)
# ──────────────────────────────────────────────

class TutorAction(BaseModel):
    """Action sent by the AI tutor to the environment."""
    action_type: str = Field(
        ...,
        description="ask_question | give_hint | explain_concept | increase_difficulty | decrease_difficulty"
    )
    content: str = Field(default="", description="Question/hint/explanation text")
    target_concept: str = Field(default="", description="Target concept")
    difficulty: int = Field(default=1, ge=1, le=5, description="Difficulty 1-5")


class TutorObservation(BaseModel):
    """Observation returned to the tutor agent."""
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
    # Required by OpenEnv serialization
    reward: float = Field(default=0.0, description="Reward from the last step")
    done: bool = Field(default=False, description="Whether the episode is done")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info")


# ──────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────

class AdaptiveTutorEnv(Environment):
    """
    OpenEnv-compatible environment for AdaptiveTutor AI.
    
    The AI tutor interacts with a simulated student, receiving
    observations about student performance and expert feedback.
    The goal is to bring the student to mastery (0.8+ overall)
    within 20 steps, while adapting to shifting expert preferences.
    """

    MAX_STEPS: int = 20
    MASTERY_THRESHOLD: float = 0.8

    def __init__(self):
        """Initialize the environment with all sub-components."""
        super().__init__()
        self.question_gen = QuestionGenerator(subjects_dir="subjects")
        self.expert_sim = ExpertSimulator()
        self.student: Optional[StudentProfile] = None
        self.current_subject: str = ""
        self.current_difficulty: int = 1
        self.step_count: int = 0
        self.last_student_correct: bool = False
        self.last_expert_feedback: Optional[str] = None
        self.last_expert_pref_changed: bool = False
        self.last_reward: float = 0.0
        self.last_reward_breakdown: Dict[str, float] = {}
        self.mastery_concepts_rewarded: set = set()
        self.episode_done: bool = False

    def reset(self) -> TutorObservation:
        """
        Reset the environment for a new tutoring episode.
        
        Initializes a fresh student with random knowledge levels,
        picks a random subject, and resets expert preferences.
        
        Returns:
            Initial TutorObservation with student profile.
        """
        # Pick a random subject
        self.current_subject = self.question_gen.get_random_subject()
        concepts = self.question_gen.get_concepts_for_subject(self.current_subject)

        # Create a new student
        self.student = StudentProfile(name="Simulated Student")
        self.student.initialize_knowledge(concepts)

        # Reset all state
        self.current_difficulty = 1
        self.step_count = 0
        self.last_student_correct = False
        self.last_expert_feedback = None
        self.last_expert_pref_changed = False
        self.last_reward = 0.0
        self.last_reward_breakdown = {}
        self.mastery_concepts_rewarded = set()
        self.episode_done = False

        # Reset sub-components
        self.question_gen.reset_asked_questions()
        self.expert_sim.reset()

        expert_name = self.question_gen.get_expert_name(self.current_subject)
        level_name = self.question_gen.get_level_name(self.current_subject, 1)
        print(f"\n[RESET] Subject: {self.current_subject} | Expert: {expert_name}")
        print(f"[RESET] Starting Level: {level_name}")
        print(f"[RESET] Student Knowledge: {self.student.knowledge_map}")

        return self._make_observation()

    def step(self, action: TutorAction) -> TutorObservation:
        """
        Process one tutoring step.
        
        Flow:
        1. Parse tutor action
        2. Get a question based on action
        3. Simulate student response
        4. Update student knowledge
        5. Get expert feedback (probabilistic)
        6. Calculate multi-factor reward
        7. Check termination conditions
        
        Args:
            action: The TutorAction from the agent.
            
        Returns:
            Updated TutorObservation.
        """
        if self.episode_done or self.student is None:
            return self._make_observation()

        self.step_count += 1

        # --- 1. Process action ---
        action_type = action.action_type
        target_concept = action.target_concept
        difficulty = action.difficulty

        # Update difficulty based on action
        if action_type == "increase_difficulty":
            self.current_difficulty = min(5, self.current_difficulty + 1)
            difficulty = self.current_difficulty
        elif action_type == "decrease_difficulty":
            self.current_difficulty = max(1, self.current_difficulty - 1)
            difficulty = self.current_difficulty
        else:
            self.current_difficulty = difficulty

        # --- 2. Get question and simulate response ---
        student_correct = False
        if action_type in ("ask_question", "increase_difficulty", "decrease_difficulty"):
            question = self.question_gen.get_question(
                self.current_subject, difficulty, target_concept
            )
            if question:
                target_concept = question.get("concept", target_concept)
                student_correct = self.student.simulate_response(target_concept, difficulty)
                self.student.update_knowledge(target_concept, student_correct)
                print(
                    f"[STEP {self.step_count}] Q: {question['question'][:60]}... "
                    f"| Correct: {student_correct} | Concept: {target_concept}"
                )
            else:
                # No question available — treat as give_hint
                action_type = "give_hint"
                print(f"[STEP {self.step_count}] No question found, treating as hint.")

        elif action_type == "give_hint":
            # Hints help slightly — simulate a small knowledge boost
            if target_concept and target_concept in self.student.knowledge_map:
                current = self.student.knowledge_map[target_concept]
                self.student.knowledge_map[target_concept] = min(1.0, current + 0.03)
            print(f"[STEP {self.step_count}] Hint given for: {target_concept}")

        elif action_type == "explain_concept":
            # Explanations help more — simulate knowledge boost
            if target_concept and target_concept in self.student.knowledge_map:
                current = self.student.knowledge_map[target_concept]
                self.student.knowledge_map[target_concept] = min(1.0, current + 0.05)
            print(f"[STEP {self.step_count}] Concept explained: {target_concept}")

        self.last_student_correct = student_correct

        # --- 3. Expert feedback ---
        expert_feedback, pref_changed = self.expert_sim.step(
            self.current_subject,
            action.model_dump()
        )
        self.last_expert_feedback = expert_feedback
        self.last_expert_pref_changed = pref_changed

        # --- 4. Calculate rewards ---
        concept_mastery = self.student.knowledge_map.get(target_concept, 0.3)
        overall_mastery = self.student.get_overall_mastery()

        # Determine if tutor adapted to expert feedback
        tutor_adapted = self._check_expert_adaptation(action, expert_feedback)

        # Check if this concept just reached mastery (one-time bonus)
        concept_already_rewarded = target_concept in self.mastery_concepts_rewarded
        if concept_mastery >= self.MASTERY_THRESHOLD and not concept_already_rewarded:
            self.mastery_concepts_rewarded.add(target_concept)

        total_reward, breakdown = RewardManager.calculate_total_reward(
            student_correct=student_correct,
            concept=target_concept if not concept_already_rewarded else "__none__",
            knowledge_map=self.student.knowledge_map,
            chosen_difficulty=difficulty,
            student_mastery=concept_mastery,
            consecutive_correct=self.student.consecutive_correct,
            consecutive_wrong=self.student.consecutive_wrong,
            expert_feedback=expert_feedback,
            tutor_adapted=tutor_adapted,
            overall_mastery=overall_mastery,
            steps_taken=self.step_count,
            max_steps=self.MAX_STEPS,
        )

        self.last_reward = total_reward
        self.last_reward_breakdown = breakdown

        print(f"[REWARD] Total: {total_reward:.2f} | Breakdown: {breakdown}")

        # --- 5. Check termination ---
        mastery_achieved = overall_mastery >= self.MASTERY_THRESHOLD
        if mastery_achieved or self.step_count >= self.MAX_STEPS:
            self.episode_done = True
            if mastery_achieved:
                print(f"[DONE] Mastery achieved in {self.step_count} steps!")
            else:
                print(f"[DONE] Max steps ({self.MAX_STEPS}) reached. Mastery: {overall_mastery:.2f}")

        return self._make_observation()

    @property
    def state(self) -> Dict[str, Any]:
        """
        Return the internal state for inspection.
        
        Returns:
            Dictionary with full environment state.
        """
        return {
            "subject": self.current_subject,
            "difficulty": self.current_difficulty,
            "step_count": self.step_count,
            "student": self.student.get_profile_summary() if self.student else {},
            "last_reward": self.last_reward,
            "reward_breakdown": self.last_reward_breakdown,
            "expert_preferences": self.expert_sim.get_current_preferences(self.current_subject),
            "episode_done": self.episode_done,
        }

    def _make_observation(self) -> TutorObservation:
        """
        Build a TutorObservation from the current state.
        
        Returns:
            TutorObservation with all current information.
        """
        profile = self.student.get_profile_summary() if self.student else {}
        overall_mastery = self.student.get_overall_mastery() if self.student else 0.0

        obs = TutorObservation(
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
            info=self.last_reward_breakdown,
        )
        return obs

    def _check_expert_adaptation(
        self, action: TutorAction, feedback: Optional[str]
    ) -> bool:
        """
        Check whether the tutor adapted to the expert's feedback.
        
        Simplified heuristic: if expert said to change approach and tutor
        changed action type or difficulty, we consider it adapted.
        
        Args:
            action: The tutor's current action.
            feedback: Expert feedback (if any).
            
        Returns:
            True if the tutor appears to have adapted.
        """
        if feedback is None:
            return False

        feedback_lower = feedback.lower()

        # Check for various adaptation signals
        if "increase difficulty" in feedback_lower and action.action_type == "increase_difficulty":
            return True
        if "decrease" in feedback_lower and action.action_type == "decrease_difficulty":
            return True
        if "different approach" in feedback_lower and action.action_type in ("explain_concept", "give_hint"):
            return True
        if "hint" in feedback_lower and action.action_type == "give_hint":
            return True
        if "specific" in feedback_lower and action.content:
            return True
        if "focus on" in feedback_lower and action.target_concept:
            return True

        # Default: 40% chance of being considered adapted
        # (the real LLM agent would do much better)
        return random.random() < 0.4


# ──────────────────────────────────────────────
#  FastAPI App Setup
# ──────────────────────────────────────────────

env_instance = AdaptiveTutorEnv()

def env_factory():
    """Factory function for OpenEnv."""
    return env_instance

app = create_fastapi_app(env_factory, TutorAction, TutorObservation)

# --- HTTPS Proxy Support (HuggingFace Spaces) ---
# HF Spaces serves over HTTPS via a reverse proxy. Without this,
# Gradio generates http:// URLs causing "Mixed Content" browser errors.
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

@app.middleware("http")
async def force_https_scheme(request, call_next):
    """Trust X-Forwarded-Proto header from HF's reverse proxy."""
    if request.headers.get("x-forwarded-proto") == "https":
        request.scope["scheme"] = "https"
    return await call_next(request)


# ──────────────────────────────────────────────
#  Gradio Dashboard
# ──────────────────────────────────────────────
import gradio as gr
from fastapi.responses import HTMLResponse
from app import demo

# Mount the rich interactive dashboard from app.py onto the root path
# so it becomes the main UI for HuggingFace Spaces.
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        forwarded_allow_ips="*",
        proxy_headers=True,
    )
