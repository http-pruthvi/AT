import json, time

class SessionManager:
    """Manages human student session state"""
    def __init__(self):
        self.reset()
    
    def reset(self, student_name="Student", subject="Math"):
        self.student_name = student_name
        self.subject = subject
        self.mastery_map = {
            "Math": {"algebra": 0.3, "geometry": 0.2, "calculus": 0.15, "statistics": 0.4},
            "Science": {"photosynthesis": 0.3, "cell_division": 0.2, "newton_laws": 0.35, "thermodynamics": 0.25},
            "History": {"ww2": 0.3, "independence": 0.4, "renaissance": 0.2, "cold_war": 0.25}
        }
        self.chat_history = []
        self.streak = 0
        self.questions_asked = 0
        self.correct_answers = 0
        self.session_start = time.time()
        self.teacher_notes = []
    
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
        return {
            "questions": self.questions_asked,
            "accuracy": self.get_accuracy(),
            "time": self.get_session_time()
        }
    
    def is_complete(self):
        concepts = self.mastery_map.get(self.subject, {})
        return all(v >= 0.8 for v in concepts.values()) or self.questions_asked >= 20
