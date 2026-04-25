"""
reward.py - Multiple Independent Reward Functions for AdaptiveTutor AI

Implements FIVE separate reward functions that are summed for the final
reward. Each function targets a different aspect of good tutoring:

1. correctness_reward: Did the student get the answer right?
2. mastery_reward: Did a concept reach mastery threshold?
3. difficulty_reward: Was the difficulty level appropriate?
4. expert_reward: Did the tutor adapt to expert feedback?
5. efficiency_reward: Was mastery achieved quickly?

Anti-hacking: Each step's total reward is capped at 5.0 to prevent
reward gaming through degenerate strategies.
"""

from typing import Dict, Optional, Tuple


class RewardManager:
    """
    Calculates rewards from multiple independent functions.
    
    Each reward function is independent — they don't share state.
    The final reward is the sum of all individual rewards, capped
    at a maximum per step to prevent reward hacking.
    """

    MAX_REWARD_PER_STEP: float = 5.0
    MIN_REWARD_PER_STEP: float = -3.0

    # --- Individual Reward Constants ---
    CORRECT_REWARD: float = 1.0
    WRONG_PENALTY: float = -0.3
    MASTERY_BONUS: float = 2.0
    DIFFICULTY_MATCH_REWARD: float = 0.5
    DIFFICULTY_MISMATCH_PENALTY: float = -0.5
    EXPERT_ADAPTED_REWARD: float = 1.5
    EXPERT_IGNORED_PENALTY: float = -1.0
    EFFICIENCY_BONUS: float = 3.0

    @classmethod
    def correctness_reward(cls, student_correct: bool) -> float:
        """
        Reward based on whether the student answered correctly.
        
        A correct answer means the tutor chose an appropriate question.
        
        Args:
            student_correct: Whether the student's answer was correct.
            
        Returns:
            +1.0 for correct, -0.3 for wrong.
        """
        if student_correct:
            return cls.CORRECT_REWARD
        return cls.WRONG_PENALTY

    @classmethod
    def mastery_reward(
        cls,
        concept: str,
        knowledge_map: Dict[str, float],
        mastery_threshold: float = 0.8
    ) -> float:
        """
        Reward when a concept reaches mastery threshold.
        
        This is a one-time bonus per concept that crosses the threshold.
        
        Args:
            concept: The concept being tested.
            knowledge_map: Student's current knowledge map.
            mastery_threshold: Threshold for mastery (default 0.8).
            
        Returns:
            +2.0 if concept just reached mastery, 0.0 otherwise.
        """
        mastery = knowledge_map.get(concept, 0.0)
        if mastery >= mastery_threshold:
            return cls.MASTERY_BONUS
        return 0.0

    @classmethod
    def difficulty_reward(
        cls,
        chosen_difficulty: int,
        student_mastery: float,
        student_correct: bool,
        consecutive_correct: int,
        consecutive_wrong: int
    ) -> float:
        """
        Reward for choosing the right difficulty level.
        
        Good tutors match difficulty to student ability:
        - If student has 3+ consecutive correct → should increase difficulty
        - If student has 2+ consecutive wrong → should decrease difficulty
        - If difficulty matches mastery level → good choice
        
        Args:
            chosen_difficulty: The difficulty level chosen (1-5).
            student_mastery: Student's mastery of the target concept (0-1).
            student_correct: Whether the student just answered correctly.
            consecutive_correct: Number of consecutive correct answers.
            consecutive_wrong: Number of consecutive wrong answers.
            
        Returns:
            +0.5 for good difficulty match, -0.5 for bad match.
        """
        # Expected difficulty based on mastery
        expected_difficulty = max(1, min(5, int(student_mastery * 5) + 1))

        # Check for adaptive difficulty behavior
        if consecutive_correct >= 3 and chosen_difficulty <= expected_difficulty:
            # Student is on a streak — tutor SHOULD be increasing difficulty
            return cls.DIFFICULTY_MISMATCH_PENALTY
        
        if consecutive_wrong >= 2 and chosen_difficulty >= expected_difficulty:
            # Student is struggling — tutor SHOULD decrease difficulty
            return cls.DIFFICULTY_MISMATCH_PENALTY

        # Check if difficulty is roughly in the right range
        diff_gap = abs(chosen_difficulty - expected_difficulty)
        if diff_gap <= 1:
            return cls.DIFFICULTY_MATCH_REWARD
        else:
            return cls.DIFFICULTY_MISMATCH_PENALTY

    @classmethod
    def expert_reward(
        cls,
        expert_feedback: Optional[str],
        tutor_adapted: bool
    ) -> float:
        """
        Reward for adapting to expert feedback.
        
        When an expert gives feedback, the tutor should adjust its
        strategy accordingly. Ignoring feedback is penalized.
        
        Args:
            expert_feedback: The expert's feedback string, or None.
            tutor_adapted: Whether the tutor adapted to the feedback.
            
        Returns:
            +1.5 if adapted, -1.0 if ignored, 0.0 if no feedback.
        """
        if expert_feedback is None:
            return 0.0
        
        if tutor_adapted:
            return cls.EXPERT_ADAPTED_REWARD
        return cls.EXPERT_IGNORED_PENALTY

    @classmethod
    def efficiency_reward(
        cls,
        overall_mastery: float,
        steps_taken: int,
        max_steps: int = 20,
        mastery_threshold: float = 0.8
    ) -> float:
        """
        Bonus for achieving full mastery efficiently (fast).
        
        Only awarded when overall mastery reaches the threshold.
        Bigger bonus for fewer steps used.
        
        Args:
            overall_mastery: Student's overall mastery score.
            steps_taken: Number of steps taken so far.
            max_steps: Maximum allowed steps.
            mastery_threshold: Required mastery threshold.
            
        Returns:
            +3.0 scaled by efficiency if mastery achieved, 0.0 otherwise.
        """
        if overall_mastery < mastery_threshold:
            return 0.0
        
        # Efficiency multiplier: more bonus for fewer steps
        efficiency = max(0.1, (max_steps - steps_taken) / max_steps)
        return cls.EFFICIENCY_BONUS * efficiency

    @classmethod
    def calculate_total_reward(
        cls,
        student_correct: bool,
        concept: str,
        knowledge_map: Dict[str, float],
        chosen_difficulty: int,
        student_mastery: float,
        consecutive_correct: int,
        consecutive_wrong: int,
        expert_feedback: Optional[str],
        tutor_adapted: bool,
        overall_mastery: float,
        steps_taken: int,
        max_steps: int = 20
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the total reward from all independent reward functions.
        
        Anti-hacking: The total reward is capped at MAX_REWARD_PER_STEP (5.0)
        and floored at MIN_REWARD_PER_STEP (-3.0) to prevent degenerate
        strategies.
        
        Args:
            All parameters for individual reward functions.
            
        Returns:
            Tuple of (capped_total_reward, breakdown_dict).
        """
        # Calculate each reward independently
        r_correct = cls.correctness_reward(student_correct)
        r_mastery = cls.mastery_reward(concept, knowledge_map)
        r_difficulty = cls.difficulty_reward(
            chosen_difficulty, student_mastery, student_correct,
            consecutive_correct, consecutive_wrong
        )
        r_expert = cls.expert_reward(expert_feedback, tutor_adapted)
        r_efficiency = cls.efficiency_reward(
            overall_mastery, steps_taken, max_steps
        )

        # Build breakdown
        breakdown = {
            "correctness": r_correct,
            "mastery": r_mastery,
            "difficulty": r_difficulty,
            "expert": r_expert,
            "efficiency": r_efficiency,
        }

        # Sum all rewards
        raw_total = sum(breakdown.values())

        # Anti-hacking: cap the total
        capped_total = max(cls.MIN_REWARD_PER_STEP, min(cls.MAX_REWARD_PER_STEP, raw_total))

        return capped_total, breakdown
