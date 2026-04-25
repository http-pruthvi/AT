"""
client.py - Pydantic Models & Agent Client for AdaptiveTutor AI

Defines typed action and observation models for the OpenEnv protocol,
plus a demonstration agent that interacts with the AdaptiveTutor server.
"""

import asyncio
import random
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import httpx


# ──────────────────────────────────────────────
#  Pydantic Models (shared between server & client)
# ──────────────────────────────────────────────

class TutorAction(BaseModel):
    """
    Action sent by the AI tutor agent to the environment.
    
    The tutor can:
    - ask_question: Present a question to the student
    - give_hint: Provide a hint for the current topic
    - explain_concept: Explain a concept before asking
    - increase_difficulty: Move to a harder level
    - decrease_difficulty: Move to an easier level
    """
    action_type: str = Field(
        ...,
        description="Type of action: ask_question, give_hint, explain_concept, increase_difficulty, decrease_difficulty"
    )
    content: str = Field(
        default="",
        description="The question text, hint text, or explanation content"
    )
    target_concept: str = Field(
        default="",
        description="The concept this action targets"
    )
    difficulty: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Difficulty level (1-5)"
    )


class TutorObservation(BaseModel):
    """
    Observation returned by the environment to the tutor agent.
    
    Contains all the information the tutor needs to decide its
    next action, including student state, expert feedback, and
    session progress.
    """
    student_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Student's knowledge map and stats"
    )
    current_topic: str = Field(
        default="",
        description="The current subject being taught"
    )
    current_difficulty: int = Field(
        default=1,
        description="Current difficulty level (1-5)"
    )
    student_correct: bool = Field(
        default=False,
        description="Whether the student answered the last question correctly"
    )
    consecutive_correct: int = Field(
        default=0,
        description="Current streak of correct answers"
    )
    consecutive_wrong: int = Field(
        default=0,
        description="Current streak of wrong answers"
    )
    expert_feedback: Optional[str] = Field(
        default=None,
        description="Feedback from the subject matter expert (if any)"
    )
    expert_preference_changed: bool = Field(
        default=False,
        description="Whether the expert's preferences just shifted"
    )
    session_progress: float = Field(
        default=0.0,
        description="Progress through the episode (0.0 to 1.0)"
    )
    mastery_achieved: bool = Field(
        default=False,
        description="Whether the student reached overall mastery (0.8+)"
    )


class TutorResponse(BaseModel):
    """
    Top-level response from the OpenEnv server.
    Wraps the observation with reward and done signal.
    """
    observation: TutorObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────
#  Demo Agent
# ──────────────────────────────────────────────

async def run_agent():
    """
    Demonstration agent that interacts with the AdaptiveTutor server.
    
    This agent uses simple heuristics:
    - If student is struggling → decrease difficulty or give hints
    - If student is on a streak → increase difficulty
    - If expert feedback received → adapt strategy
    - Otherwise → ask questions targeting weak concepts
    """
    base_url = "http://127.0.0.1:7860"
    print(f"[*] Connecting to AdaptiveTutor AI at {base_url}...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Reset Environment
        try:
            print("[*] Resetting environment...")
            response = await client.post(f"{base_url}/reset")
            res = TutorResponse(**response.json())
            obs = res.observation
            done = res.done
        except Exception as e:
            print(f"[!] Error: Could not connect to server. Is it running? {e}")
            return

        print(f"[+] Session Started: {obs.current_topic.upper()} (Level {obs.current_difficulty})")
        print(f"[+] Student Profile: {obs.student_profile}")

        step_count = 0
        total_reward = 0.0
        last_expert_feedback = None

        while not done:
            step_count += 1
            print(f"\n{'='*50}")
            print(f"--- Step {step_count} ---")

            # 2. Decide Action based on heuristics
            action = _decide_action(obs, last_expert_feedback)
            last_expert_feedback = obs.expert_feedback

            print(f"[>] Action: {action.action_type} | Concept: {action.target_concept} | Difficulty: {action.difficulty}")
            if action.content:
                print(f"[>] Content: {action.content[:80]}...")

            # 3. Perform Step (OpenEnv expects action nested under "action" key)
            response = await client.post(
                f"{base_url}/step",
                json={"action": action.model_dump()}
            )
            res = TutorResponse(**response.json())
            obs = res.observation
            done = res.done

            total_reward += res.reward
            print(f"[<] Correct: {obs.student_correct} | Reward: {res.reward:.2f} | Total: {total_reward:.2f}")
            print(f"[<] Mastery: {obs.student_profile.get('overall_mastery', 0):.2f} | Progress: {obs.session_progress:.1%}")
            
            if obs.expert_feedback:
                print(f"[EXPERT] {obs.expert_feedback}")
            if obs.expert_preference_changed:
                print("[EXPERT] ** Expert preferences have SHIFTED! **")
            if obs.mastery_achieved:
                print("[WIN] Student has achieved mastery!")

            await asyncio.sleep(0.3)

        print(f"\n{'='*50}")
        print(f"[!] Episode Finished! Steps: {step_count} | Total Reward: {total_reward:.2f}")
        print(f"[!] Mastery Achieved: {obs.mastery_achieved}")


def _decide_action(obs: TutorObservation, last_feedback: Optional[str]) -> TutorAction:
    """
    Simple heuristic agent for demonstration.
    
    Args:
        obs: Current observation from the environment.
        last_feedback: Previous expert feedback (if any).
        
    Returns:
        A TutorAction to send to the environment.
    """
    profile = obs.student_profile
    weak = profile.get("weak_concepts", [])
    weakest = profile.get("weakest_concept", "")

    # Rule 1: If student is struggling, decrease difficulty or give hint
    if obs.consecutive_wrong >= 2:
        if obs.current_difficulty > 1:
            return TutorAction(
                action_type="decrease_difficulty",
                content="Let's try something a bit easier.",
                target_concept=weakest or obs.current_topic,
                difficulty=max(1, obs.current_difficulty - 1),
            )
        else:
            return TutorAction(
                action_type="give_hint",
                content="Here's a hint to help you.",
                target_concept=weakest or obs.current_topic,
                difficulty=obs.current_difficulty,
            )

    # Rule 2: If student is on a streak, increase difficulty
    if obs.consecutive_correct >= 3:
        return TutorAction(
            action_type="increase_difficulty",
            content="Great job! Let's try something harder.",
            target_concept=weakest or obs.current_topic,
            difficulty=min(5, obs.current_difficulty + 1),
        )

    # Rule 3: If expert gave feedback, adapt (simplified)
    if last_feedback and "different approach" in last_feedback.lower():
        return TutorAction(
            action_type="explain_concept",
            content="Let me explain this concept differently.",
            target_concept=weakest or obs.current_topic,
            difficulty=obs.current_difficulty,
        )

    # Default: Ask a question targeting the weakest concept
    target = weakest if weakest else obs.current_topic
    return TutorAction(
        action_type="ask_question",
        content="",
        target_concept=target,
        difficulty=obs.current_difficulty,
    )


if __name__ == "__main__":
    asyncio.run(run_agent())
