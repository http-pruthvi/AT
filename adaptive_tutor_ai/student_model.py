"""
student_model.py - Simulated Student for AdaptiveTutor AI

Simulates a realistic student with a knowledge map across concepts.
The student has probabilistic response behavior based on mastery level
and question difficulty, creating a challenging environment for the
AI tutor to navigate.
"""

import random
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    """
    Represents a simulated student's knowledge state.
    
    Each concept in the knowledge_map has a mastery score from 0.0 to 1.0.
    The student's responses are probabilistic, based on their mastery
    of the target concept and the difficulty of the question.
    """
    name: str = Field(default="Simulated Student", description="Student name")
    knowledge_map: Dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of concept -> mastery score (0.0 to 1.0)"
    )
    total_questions_answered: int = Field(default=0, description="Total questions attempted")
    total_correct: int = Field(default=0, description="Total correct answers")
    consecutive_correct: int = Field(default=0, description="Current streak of correct answers")
    consecutive_wrong: int = Field(default=0, description="Current streak of wrong answers")
    current_difficulty: int = Field(default=1, description="Current difficulty level (1-5)")
    weak_concepts: List[str] = Field(default_factory=list, description="Concepts with mastery < 0.4")
    strong_concepts: List[str] = Field(default_factory=list, description="Concepts with mastery >= 0.7")

    def initialize_knowledge(self, concepts: List[str]) -> None:
        """
        Initialize the knowledge map with random starting mastery levels.
        
        Creates a realistic distribution: some concepts strong, some weak,
        most in the middle. This forces the tutor to identify and address
        specific weaknesses.
        
        Args:
            concepts: List of concept names to initialize.
        """
        for concept in concepts:
            # Bimodal-ish distribution: some strong, some weak
            roll = random.random()
            if roll < 0.25:
                # Weak area (25% of concepts)
                mastery = random.uniform(0.1, 0.35)
            elif roll < 0.60:
                # Medium area (35% of concepts)
                mastery = random.uniform(0.35, 0.6)
            else:
                # Strong area (40% of concepts)
                mastery = random.uniform(0.6, 0.85)
            self.knowledge_map[concept] = round(mastery, 2)
        
        self._update_concept_lists()

    def simulate_response(self, concept: str, difficulty: int) -> bool:
        """
        Simulate whether the student answers correctly.
        
        The probability of a correct answer depends on:
        1. Student's mastery of the concept
        2. The difficulty level of the question
        3. Random noise (students are unpredictable!)
        
        Probabilities:
        - Weak area (mastery < 0.4): 40% base correct rate
        - Strong area (mastery >= 0.7): 85% base correct rate
        - Medium area: 65% base correct rate
        - Difficulty too high (diff > mastery*5 + 1): 20% correct rate
        - Matched difficulty: 65% correct rate
        
        Args:
            concept: The concept being tested.
            difficulty: Difficulty level (1-5).
            
        Returns:
            True if the student answers correctly, False otherwise.
        """
        mastery = self.knowledge_map.get(concept, 0.3)
        
        # Base probability from mastery level
        if mastery < 0.4:
            base_prob = 0.40
        elif mastery >= 0.7:
            base_prob = 0.85
        else:
            base_prob = 0.65
        
        # Difficulty adjustment
        # Student's "effective level" is roughly mastery * 5
        effective_level = mastery * 5.0
        difficulty_gap = difficulty - effective_level
        
        if difficulty_gap > 1.5:
            # Question is way too hard for this student
            base_prob = 0.20
        elif difficulty_gap > 0.5:
            # Question is somewhat hard
            base_prob *= 0.7
        elif difficulty_gap < -1.0:
            # Question is too easy — student gets it easily
            base_prob = min(0.95, base_prob + 0.15)
        
        # Add small random noise for realism
        noise = random.uniform(-0.05, 0.05)
        final_prob = max(0.05, min(0.95, base_prob + noise))
        
        # Roll the dice
        is_correct = random.random() < final_prob
        
        # Update tracking
        self.total_questions_answered += 1
        if is_correct:
            self.total_correct += 1
            self.consecutive_correct += 1
            self.consecutive_wrong = 0
        else:
            self.consecutive_wrong += 1
            self.consecutive_correct = 0
        
        return is_correct

    def update_knowledge(self, concept: str, correct: bool) -> None:
        """
        Update the student's knowledge based on whether they got the answer right.
        
        Learning dynamics:
        - Correct answer: +0.1 mastery (learning happens)
        - Wrong answer: -0.05 mastery (slight forgetting/confusion)
        - Mastery is clamped between 0.0 and 1.0
        
        Args:
            concept: The concept that was tested.
            correct: Whether the student answered correctly.
        """
        current = self.knowledge_map.get(concept, 0.3)
        
        if correct:
            new_mastery = min(1.0, current + 0.1)
        else:
            new_mastery = max(0.0, current - 0.05)
        
        self.knowledge_map[concept] = round(new_mastery, 2)
        self._update_concept_lists()

    def get_overall_mastery(self) -> float:
        """
        Calculate the student's average mastery across all concepts.
        
        Returns:
            Average mastery score (0.0 to 1.0).
        """
        if not self.knowledge_map:
            return 0.0
        return round(sum(self.knowledge_map.values()) / len(self.knowledge_map), 3)

    def get_weakest_concept(self) -> Optional[str]:
        """
        Find the concept with the lowest mastery.
        
        Returns:
            Name of the weakest concept, or None if knowledge map is empty.
        """
        if not self.knowledge_map:
            return None
        return min(self.knowledge_map, key=self.knowledge_map.get)

    def get_profile_summary(self) -> Dict:
        """
        Get a summary of the student's current state for the tutor.
        
        Returns:
            Dictionary with student profile information.
        """
        return {
            "overall_mastery": self.get_overall_mastery(),
            "total_answered": self.total_questions_answered,
            "accuracy": round(self.total_correct / max(1, self.total_questions_answered), 2),
            "consecutive_correct": self.consecutive_correct,
            "consecutive_wrong": self.consecutive_wrong,
            "weak_concepts": self.weak_concepts,
            "strong_concepts": self.strong_concepts,
            "weakest_concept": self.get_weakest_concept(),
            "knowledge_map": dict(self.knowledge_map),
        }

    def _update_concept_lists(self) -> None:
        """Update the weak and strong concept lists based on current mastery."""
        self.weak_concepts = [
            c for c, m in self.knowledge_map.items() if m < 0.4
        ]
        self.strong_concepts = [
            c for c, m in self.knowledge_map.items() if m >= 0.7
        ]
