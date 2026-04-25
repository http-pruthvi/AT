import json
import os
import random
import time

class SessionManager:
    """Manages human student session state"""
    def __init__(self):
        self.profiles_dir = "profiles"
        self.logs_dir = "logs"
        os.makedirs(self.profiles_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.reset()
    
    def reset(self, student_name="Student", subject="Math"):
        self.student_name = student_name
        self.subject = subject
        self.subject_key = (subject or "").strip().lower()
        self.mastery_map = self._build_mastery_map()
        self.chat_history = []
        self.streak = 0
        self.questions_asked = 0
        self.correct_answers = 0
        self.session_start = time.time()
        self.teacher_notes = []
        self.initial_average_mastery = self._average_mastery()

    def _build_mastery_map(self):
        """Build concept mastery from actual subject files to avoid drift."""
        default_mastery = {
            "Math": {"addition": 0.3, "subtraction": 0.3, "multiplication": 0.3},
            "Science": {"cells": 0.3, "forces": 0.3, "energy": 0.3},
            "History": {"timeline": 0.3, "causes": 0.3, "effects": 0.3},
        }

        subject_file = os.path.join("subjects", f"{self.subject_key}.json")
        if not os.path.exists(subject_file):
            return default_mastery

        try:
            with open(subject_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            concepts = set()
            for level_data in data.get("levels", {}).values():
                for concept in level_data.get("concepts", []):
                    concepts.add(concept)

            concept_map = {
                concept: round(random.uniform(0.2, 0.45), 2)
                for concept in concepts
            }
            # Keep UI subject labels in title case while using file-backed concepts.
            return {
                "Math": concept_map if self.subject_key == "math" else default_mastery["Math"],
                "Science": concept_map if self.subject_key == "science" else default_mastery["Science"],
                "History": concept_map if self.subject_key == "history" else default_mastery["History"],
            }
        except Exception:
            return default_mastery
    
    def get_weak_concepts(self):
        concepts = self.mastery_map.get(self.subject, {})
        return sorted(concepts.items(), key=lambda x: x[1])[:3]
    
    def update_mastery(self, concept, delta):
        if self.subject in self.mastery_map and concept in self.mastery_map[self.subject]:
            current = self.mastery_map[self.subject][concept]
            self.mastery_map[self.subject][concept] = max(0.0, min(1.0, current + delta))
    
    def add_message(self, role, content, is_correct=None):
        self.chat_history.append({"role": role, "content": content, "timestamp": time.time(), "is_correct": is_correct})
    
    def add_teacher_note(self, note):
        self.teacher_notes.append({"note": note, "timestamp": time.time()})
    
    def get_active_teacher_note(self):
        if self.teacher_notes and (time.time() - self.teacher_notes[-1]["timestamp"] < 300):
            return self.teacher_notes[-1]["note"]
        return None
    
    def get_accuracy(self):
        return int((self.correct_answers / self.questions_asked) * 100) if self.questions_asked > 0 else 0
    
    def get_session_time(self):
        elapsed = int(time.time() - self.session_start)
        return f"{elapsed // 60}:{elapsed % 60:02d}"
    
    def get_summary(self):
        metrics = self.get_learning_metrics()
        return {
            "questions": self.questions_asked,
            "accuracy": self.get_accuracy(),
            "time": self.get_session_time(),
            "learning_gain_pct": metrics["learning_gain_pct"],
            "mastery_now_pct": metrics["mastery_now_pct"],
        }
    
    def is_complete(self):
        concepts = self.mastery_map.get(self.subject, {})
        return all(v >= 0.8 for v in concepts.values()) or self.questions_asked >= 20

    def _average_mastery(self):
        concepts = self.mastery_map.get(self.subject, {})
        if not concepts:
            return 0.0
        return sum(concepts.values()) / len(concepts)

    def _profile_path(self):
        safe_name = "".join(c.lower() if c.isalnum() else "_" for c in self.student_name).strip("_")
        if not safe_name:
            safe_name = "student"
        return os.path.join(self.profiles_dir, f"{safe_name}.json")

    def load_profile(self):
        """Load persisted profile if present for this student+subject."""
        path = self._profile_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_subject = data.get("subject")
            if saved_subject != self.subject:
                return False
            saved_mastery = data.get("mastery_map", {}).get(self.subject, {})
            if not saved_mastery:
                return False
            self.mastery_map[self.subject] = saved_mastery
            self.initial_average_mastery = self._average_mastery()
            return True
        except Exception:
            return False

    def save_profile(self):
        """Persist current profile progress to disk."""
        history = []
        existing_path = self._profile_path()
        if os.path.exists(existing_path):
            try:
                with open(existing_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                history = existing.get("history", [])
            except Exception:
                history = []

        history.append({
            "timestamp": int(time.time()),
            "subject": self.subject,
            "questions": self.questions_asked,
            "accuracy": self.get_accuracy(),
            "session_time": self.get_session_time(),
            "metrics": self.get_learning_metrics(),
        })

        payload = {
            "student_name": self.student_name,
            "subject": self.subject,
            "mastery_map": self.mastery_map,
            "questions_asked": self.questions_asked,
            "correct_answers": self.correct_answers,
            "updated_at": int(time.time()),
            "history": history[-200:],
        }
        try:
            with open(self._profile_path(), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return True
        except Exception:
            return False

    def get_learning_metrics(self):
        current = self._average_mastery()
        gain = current - self.initial_average_mastery
        return {
            "mastery_start_pct": int(self.initial_average_mastery * 100),
            "mastery_now_pct": int(current * 100),
            "learning_gain_pct": round(gain * 100, 1),
        }

    def _interaction_log_path(self):
        safe_name = "".join(c.lower() if c.isalnum() else "_" for c in self.student_name).strip("_")
        if not safe_name:
            safe_name = "student"
        return os.path.join(self.logs_dir, f"{safe_name}.jsonl")

    def log_interaction(self, question, correct_answer, student_answer, is_correct, concept, difficulty):
        """Append one tutor-student turn to a JSONL log for self-improvement."""
        record = {
            "timestamp": int(time.time()),
            "student_name": self.student_name,
            "subject": self.subject,
            "question": question,
            "correct_answer": correct_answer,
            "student_answer": student_answer,
            "is_correct": bool(is_correct),
            "concept": concept,
            "difficulty": difficulty,
            "accuracy_pct": self.get_accuracy(),
            "learning_metrics": self.get_learning_metrics(),
        }
        try:
            with open(self._interaction_log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            return True
        except Exception:
            return False
