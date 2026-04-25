"""
expert_simulator.py - Snorkel AI Bonus: Shifting Expert Feedback

This is the KEY INNOVATION for the hackathon. Simulates subject matter
experts whose pedagogical preferences SHIFT over time, forcing the
AI tutor to continuously adapt its teaching strategy.

Three expert personas with distinct teaching philosophies:
- Dr. Sharma (Math): Prefers proof-based, rigorous questioning
- Ms. Patel (Science): Prefers experimental, hands-on approach
- Prof. Khan (History): Prefers primary source analysis

Expert preferences shift every 5 steps, simulating real-world scenarios
where educational standards, curricula, and best practices evolve.
"""

import random
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ExpertPersona(BaseModel):
    """A subject matter expert with shifting pedagogical preferences."""
    name: str = Field(..., description="Expert's name")
    subject: str = Field(..., description="Expert's subject area")
    preferences: List[str] = Field(
        default_factory=list,
        description="Current teaching preferences"
    )
    preference_phase: int = Field(
        default=0,
        description="Current preference phase (shifts every 5 steps)"
    )

    # All possible preference sets for this expert
    preference_phases: List[List[str]] = Field(
        default_factory=list,
        description="All possible preference phase configurations"
    )

    feedback_templates: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Feedback templates keyed by preference type"
    )


# Pre-defined expert configurations
EXPERT_CONFIGS: Dict[str, Dict] = {
    "math": {
        "name": "Dr. Sharma",
        "subject": "math",
        "preference_phases": [
            # Phase 0: Proof-based approach
            ["proof_based", "step_by_step", "formal_notation"],
            # Phase 1: Intuition-first approach (SHIFT!)
            ["visual_intuition", "real_world_examples", "estimation"],
            # Phase 2: Problem-solving focused (SHIFT!)
            ["problem_solving", "multiple_methods", "timed_practice"],
            # Phase 3: Conceptual depth (SHIFT!)
            ["proof_based", "historical_context", "connections"],
        ],
        "feedback_templates": {
            "proof_based": [
                "Excellent! But ask the student to prove WHY, not just solve.",
                "Good question, but add a proof component for deeper understanding.",
                "The student should justify each step formally.",
            ],
            "step_by_step": [
                "Break this into smaller steps — the student needs scaffolding.",
                "Show the intermediate steps, don't skip ahead.",
            ],
            "visual_intuition": [
                "Try using a visual or graphical representation instead.",
                "Can you frame this as a real-world scenario the student can picture?",
                "Abstract notation is fine later — start with intuition.",
            ],
            "real_world_examples": [
                "Connect this to something the student encounters daily.",
                "Use a concrete example before the abstract formula.",
            ],
            "problem_solving": [
                "Good, but challenge them to solve it a SECOND way.",
                "Focus on problem-solving strategy, not just the answer.",
                "Time pressure helps — consider adding urgency.",
            ],
            "multiple_methods": [
                "Show at least two different approaches to this problem.",
                "Can the student discover an alternative solution path?",
            ],
            "formal_notation": [
                "Use proper mathematical notation in the question.",
                "Ensure the student writes the answer in standard form.",
            ],
            "estimation": [
                "Before solving exactly, ask the student to estimate.",
                "Ballpark answers build mathematical intuition.",
            ],
            "timed_practice": [
                "Speed matters for this skill — push for faster recall.",
                "This type of problem should eventually be automatic.",
            ],
            "historical_context": [
                "Mention who discovered this and why it mattered.",
                "Connect this concept to its historical development.",
            ],
            "connections": [
                "Link this concept to what we covered earlier.",
                "Show how this connects to the bigger picture.",
            ],
        },
    },
    "science": {
        "name": "Ms. Patel",
        "subject": "science",
        "preference_phases": [
            # Phase 0: Experimental approach
            ["experimental", "hypothesis_driven", "data_analysis"],
            # Phase 1: Conceptual understanding (SHIFT!)
            ["conceptual", "analogies", "misconception_busting"],
            # Phase 2: Inquiry-based (SHIFT!)
            ["inquiry_based", "student_led", "open_ended"],
            # Phase 3: Application-focused (SHIFT!)
            ["experimental", "real_world_application", "interdisciplinary"],
        ],
        "feedback_templates": {
            "experimental": [
                "Frame this as an experiment the student could actually do.",
                "What would happen if we tested this in a lab?",
                "Ask the student to predict before explaining.",
            ],
            "hypothesis_driven": [
                "Have the student form a hypothesis first.",
                "Good question, but start with 'What do you think will happen?'",
            ],
            "data_analysis": [
                "Include some data for the student to interpret.",
                "Can you add a graph or table for analysis?",
            ],
            "conceptual": [
                "Focus on the underlying concept, not just the formula.",
                "Make sure the student understands WHY, not just HOW.",
            ],
            "analogies": [
                "Use an analogy to make this more accessible.",
                "Compare this to something familiar.",
            ],
            "misconception_busting": [
                "Address the common misconception about this topic.",
                "What do students usually get wrong here? Target that.",
            ],
            "inquiry_based": [
                "Let the student discover this through guided questions.",
                "Don't give the answer — lead them to it.",
            ],
            "student_led": [
                "Ask the student what THEY want to explore about this.",
                "Give the student agency in directing the learning.",
            ],
            "open_ended": [
                "Make the question open-ended — there's no single right answer.",
                "Encourage creative thinking and multiple valid approaches.",
            ],
            "real_world_application": [
                "How does this apply in the real world?",
                "Give a practical application of this concept.",
            ],
            "interdisciplinary": [
                "Connect this to math or history for a richer understanding.",
                "This concept appears in other fields too — mention that.",
            ],
        },
    },
    "history": {
        "name": "Prof. Khan",
        "subject": "history",
        "preference_phases": [
            # Phase 0: Primary source analysis
            ["primary_sources", "document_analysis", "evidence_based"],
            # Phase 1: Narrative and storytelling (SHIFT!)
            ["narrative", "empathy", "oral_history"],
            # Phase 2: Debate and argumentation (SHIFT!)
            ["debate", "multiple_perspectives", "thesis_defense"],
            # Phase 3: Global and comparative (SHIFT!)
            ["primary_sources", "comparative", "global_connections"],
        ],
        "feedback_templates": {
            "primary_sources": [
                "Include a primary source excerpt for the student to analyze.",
                "Ask 'What does this document tell us?' not just 'What happened?'",
                "Source analysis is key — who wrote it, when, and why?",
            ],
            "document_analysis": [
                "Have the student evaluate the reliability of a source.",
                "Ask about bias and perspective in historical documents.",
            ],
            "evidence_based": [
                "The student should cite specific evidence for their answer.",
                "Push for evidence, not just opinion.",
            ],
            "narrative": [
                "Tell this as a story — make it come alive for the student.",
                "Who are the people involved? Make them real.",
            ],
            "empathy": [
                "Ask the student to imagine being a person in this time period.",
                "How would YOU feel if you lived through this event?",
            ],
            "oral_history": [
                "Consider what voices are missing from the official record.",
                "How might ordinary people have experienced this differently?",
            ],
            "debate": [
                "Set up a debate — have the student argue BOTH sides.",
                "Challenge the student's first answer with a counterargument.",
            ],
            "multiple_perspectives": [
                "Present at least two perspectives on this event.",
                "Whose perspective is missing from this account?",
            ],
            "thesis_defense": [
                "Have the student formulate a thesis and defend it.",
                "Push the student to take a strong, defensible position.",
            ],
            "comparative": [
                "Compare this event to a similar one in another culture.",
                "Are there patterns across civilizations? Find them.",
            ],
            "global_connections": [
                "How did this event affect other parts of the world?",
                "Think globally — what were the ripple effects?",
            ],
        },
    },
}


class ExpertSimulator:
    """
    Simulates subject matter experts with shifting preferences.
    
    This is the Snorkel AI bonus innovation: expert preferences
    change over time, and the AI tutor must detect and adapt to
    these shifts. This simulates real-world scenarios where
    educational standards and best practices evolve.
    """

    def __init__(self):
        """Initialize the expert simulator with all expert personas."""
        self.experts: Dict[str, ExpertPersona] = {}
        self.step_count: int = 0
        self.shift_interval: int = 5  # Preferences shift every 5 steps
        self.feedback_probability: float = 0.30  # 30% chance per step
        self.last_feedback: Optional[str] = None
        self.last_preference_changed: bool = False
        self._initialize_experts()

    def _initialize_experts(self) -> None:
        """Create expert personas from the configuration."""
        for subject, config in EXPERT_CONFIGS.items():
            persona = ExpertPersona(
                name=config["name"],
                subject=subject,
                preferences=config["preference_phases"][0],
                preference_phase=0,
                preference_phases=config["preference_phases"],
                feedback_templates=config["feedback_templates"],
            )
            self.experts[subject] = persona

    def get_expert(self, subject: str) -> Optional[ExpertPersona]:
        """
        Get the expert persona for a given subject.
        
        Args:
            subject: The subject name.
            
        Returns:
            The ExpertPersona, or None if not found.
        """
        return self.experts.get(subject)

    def step(self, subject: str, tutor_action: Dict) -> Tuple[Optional[str], bool]:
        """
        Process one step — potentially generate feedback and shift preferences.
        
        Args:
            subject: The current subject.
            tutor_action: The tutor's action dict.
            
        Returns:
            Tuple of (feedback_string_or_None, preference_changed_bool).
        """
        self.step_count += 1
        expert = self.experts.get(subject)
        if not expert:
            return None, False

        # Check for preference shift
        preference_changed = False
        if self.step_count % self.shift_interval == 0:
            preference_changed = self._shift_preferences(subject)

        # Generate feedback with probability
        feedback = None
        if random.random() < self.feedback_probability:
            feedback = self._generate_feedback(subject, tutor_action)

        # Also generate context-sensitive feedback
        if not feedback:
            feedback = self._generate_contextual_feedback(subject, tutor_action)

        self.last_feedback = feedback
        self.last_preference_changed = preference_changed
        return feedback, preference_changed

    def _shift_preferences(self, subject: str) -> bool:
        """
        Shift the expert's preferences to the next phase.
        
        Args:
            subject: The subject whose expert preferences to shift.
            
        Returns:
            True if preferences changed, False otherwise.
        """
        expert = self.experts.get(subject)
        if not expert:
            return False

        old_phase = expert.preference_phase
        new_phase = (old_phase + 1) % len(expert.preference_phases)
        expert.preference_phase = new_phase
        expert.preferences = expert.preference_phases[new_phase]

        print(
            f"[EXPERT SHIFT] {expert.name}: Phase {old_phase} -> {new_phase} "
            f"| New preferences: {expert.preferences}"
        )
        return True

    def _generate_feedback(self, subject: str, tutor_action: Dict) -> Optional[str]:
        """
        Generate feedback based on current preferences.
        
        Args:
            subject: The current subject.
            tutor_action: The tutor's action dict.
            
        Returns:
            Feedback string or None.
        """
        expert = self.experts.get(subject)
        if not expert:
            return None

        # Pick a random current preference and give feedback from it
        pref = random.choice(expert.preferences)
        templates = expert.feedback_templates.get(pref, [])
        if templates:
            return random.choice(templates)
        return None

    def _generate_contextual_feedback(
        self, subject: str, tutor_action: Dict
    ) -> Optional[str]:
        """
        Generate feedback based on tutor action context.
        
        Provides specific feedback like:
        - "Question too ambiguous, be more specific"
        - "Increase difficulty now, student is ready"
        - "Student struggling, try different approach"
        - "Focus on concept X, student is weak there"
        
        Args:
            subject: The current subject.
            tutor_action: The tutor's action dict.
            
        Returns:
            Contextual feedback string or None.
        """
        action_type = tutor_action.get("action_type", "")

        # 15% chance of contextual feedback
        if random.random() > 0.15:
            return None

        contextual_templates = [
            "Question too ambiguous, be more specific.",
            "Increase difficulty now, student is ready.",
            "Student struggling, try a different approach.",
            f"Focus on concept '{tutor_action.get('target_concept', 'unknown')}', student is weak there.",
            "Good pacing, keep this difficulty level for now.",
            "Consider giving a hint before the next question.",
            "The student needs more practice at this level.",
            "Try connecting this to a real-world example.",
        ]

        return random.choice(contextual_templates)

    def get_current_preferences(self, subject: str) -> List[str]:
        """
        Get the current preference list for a subject's expert.
        
        Args:
            subject: The subject name.
            
        Returns:
            List of current preference strings.
        """
        expert = self.experts.get(subject)
        if expert:
            return expert.preferences
        return []

    def reset(self) -> None:
        """Reset the expert simulator for a new episode."""
        self.step_count = 0
        self.last_feedback = None
        self.last_preference_changed = False
        for expert in self.experts.values():
            expert.preference_phase = 0
            expert.preferences = expert.preference_phases[0]
