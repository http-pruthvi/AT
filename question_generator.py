"""
question_generator.py - Question Bank Manager for AdaptiveTutor AI

Loads questions from JSON files and serves them based on subject,
difficulty level, and target concept. Supports 3 subjects (math,
science, history) with 5 difficulty levels each.
"""

import os
import json
import random
from typing import Dict, List, Optional, Any


class QuestionGenerator:
    """
    Manages question banks loaded from JSON files.
    
    Provides questions filtered by subject, difficulty level,
    and target concept. Tracks which questions have been asked
    to avoid repetition within an episode.
    """

    def __init__(self, subjects_dir: str = "subjects"):
        """
        Initialize the question generator.
        
        Args:
            subjects_dir: Path to directory containing subject JSON files.
        """
        self.subjects_dir = subjects_dir
        self.question_banks: Dict[str, Any] = {}
        self.asked_questions: set = set()
        self._load_all_subjects()

    def _load_all_subjects(self) -> None:
        """Load all subject JSON files from the subjects directory."""
        if not os.path.exists(self.subjects_dir):
            print(f"[WARNING] Subjects directory '{self.subjects_dir}' not found.")
            return

        for filename in os.listdir(self.subjects_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.subjects_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        subject_name = data.get("subject", filename.replace(".json", ""))
                        self.question_banks[subject_name] = data
                        level_count = len(data.get("levels", {}))
                        q_count = sum(
                            len(lvl.get("questions", []))
                            for lvl in data.get("levels", {}).values()
                        )
                        print(f"[LOADED] {subject_name}: {level_count} levels, {q_count} questions")
                except Exception as e:
                    print(f"[ERROR] Failed to load {filepath}: {e}")

    def get_available_subjects(self) -> List[str]:
        """
        Get list of all available subjects.
        
        Returns:
            List of subject names.
        """
        return list(self.question_banks.keys())

    def get_random_subject(self) -> str:
        """
        Pick a random subject from available subjects.
        
        Returns:
            A random subject name.
        """
        subjects = self.get_available_subjects()
        if not subjects:
            raise ValueError("No subjects loaded. Check your subjects directory.")
        return random.choice(subjects)

    def get_concepts_for_subject(self, subject: str) -> List[str]:
        """
        Get all unique concepts for a given subject across all levels.
        
        Args:
            subject: The subject name.
            
        Returns:
            List of unique concept strings.
        """
        concepts = set()
        bank = self.question_banks.get(subject, {})
        for level_data in bank.get("levels", {}).values():
            concepts.update(level_data.get("concepts", []))
        return list(concepts)

    def get_question(
        self,
        subject: str,
        difficulty: int,
        target_concept: Optional[str] = None,
        avoid_repeats: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get a question matching the specified criteria.
        
        Args:
            subject: The subject to get a question from.
            difficulty: Difficulty level (1-5).
            target_concept: Optional specific concept to target.
            avoid_repeats: If True, skip questions already asked this episode.
            
        Returns:
            A question dict with id, question, answer, concept, hint.
            None if no matching question found.
        """
        bank = self.question_banks.get(subject)
        if not bank:
            return None

        level_key = str(difficulty)
        level_data = bank.get("levels", {}).get(level_key)
        if not level_data:
            return None

        questions = level_data.get("questions", [])

        # Filter by concept if specified
        if target_concept:
            concept_questions = [q for q in questions if q.get("concept") == target_concept]
            if concept_questions:
                questions = concept_questions
            # else: use all questions at this difficulty level (concept not available here)

        # Filter out already-asked questions
        if avoid_repeats:
            fresh_questions = [q for q in questions if q["id"] not in self.asked_questions]
            if fresh_questions:
                questions = fresh_questions
            # else: allow repeats — better than returning nothing

        if not questions:
            return None

        selected = random.choice(questions)
        self.asked_questions.add(selected["id"])
        return selected

    def generate_question(
        self,
        subject: str,
        difficulty: Any = "medium",
        concept: Optional[str] = None,
        teacher_note: Optional[str] = None,
        avoid_repeats: bool = True
    ) -> Dict[str, Any]:
        """
        Gradio-friendly wrapper for get_question.
        
        Args:
            subject: The subject.
            difficulty: String ('easy', 'medium', 'hard') or int (1-5).
            concept: Optional concept filter.
            teacher_note: Optional note (currently logged but not affecting retrieval).
            avoid_repeats: Whether to avoid repeats.
            
        Returns:
            Question dictionary.
        """
        # Map string difficulty to numeric
        diff_val = 3
        if isinstance(difficulty, str):
            mapping = {"easy": 1, "medium": 3, "hard": 5}
            diff_val = mapping.get(difficulty.lower(), 3)
        elif isinstance(difficulty, int):
            diff_val = max(1, min(5, difficulty))

        question = self.get_question(subject, diff_val, concept, avoid_repeats)
        
        if not question:
            # Fallback: try any difficulty
            for d in [3, 2, 4, 1, 5]:
                question = self.get_question(subject, d, None, avoid_repeats)
                if question: break
        
        if not question:
            return {
                "id": "fallback",
                "question": f"Can you tell me something interesting about {subject}?",
                "correct_answer": "yes",
                "concept": concept or "general",
                "hint": "Just share what you know!"
            }
            
        return question

    def get_level_name(self, subject: str, difficulty: int) -> str:
        """
        Get the human-readable name for a difficulty level.
        
        Args:
            subject: The subject name.
            difficulty: Difficulty level (1-5).
            
        Returns:
            Level name string.
        """
        bank = self.question_banks.get(subject, {})
        level_data = bank.get("levels", {}).get(str(difficulty), {})
        return level_data.get("name", f"Level {difficulty}")

    def get_expert_name(self, subject: str) -> str:
        """
        Get the expert name associated with a subject.
        
        Args:
            subject: The subject name.
            
        Returns:
            Expert name string.
        """
        bank = self.question_banks.get(subject, {})
        return bank.get("expert", "Unknown Expert")

    def reset_asked_questions(self) -> None:
        """Reset the tracking of asked questions for a new episode."""
        self.asked_questions.clear()
