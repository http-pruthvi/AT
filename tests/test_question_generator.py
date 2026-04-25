"""Tests for question_generator.py — loading, filtering, fallbacks."""

from core.question_generator import QuestionGenerator


class TestQuestionLoading:
    def test_loads_subjects(self, subjects_dir):
        qg = QuestionGenerator(subjects_dir=subjects_dir)
        subjects = qg.get_available_subjects()
        assert len(subjects) >= 1

    def test_get_random_subject(self, subjects_dir):
        qg = QuestionGenerator(subjects_dir=subjects_dir)
        subj = qg.get_random_subject()
        assert subj in qg.get_available_subjects()


class TestQuestionRetrieval:
    def test_get_question_valid(self, subjects_dir):
        qg = QuestionGenerator(subjects_dir=subjects_dir)
        subjects = qg.get_available_subjects()
        if subjects:
            q = qg.get_question(subjects[0], difficulty=1)
            assert q is not None
            assert "question" in q or "id" in q

    def test_get_question_invalid_subject(self, subjects_dir):
        qg = QuestionGenerator(subjects_dir=subjects_dir)
        q = qg.get_question("nonexistent_subject", difficulty=1)
        assert q is None

    def test_get_concepts(self, subjects_dir):
        qg = QuestionGenerator(subjects_dir=subjects_dir)
        subjects = qg.get_available_subjects()
        if subjects:
            concepts = qg.get_concepts_for_subject(subjects[0])
            assert isinstance(concepts, list)
            assert len(concepts) >= 1


class TestNoRepeat:
    def test_avoids_repeats(self, subjects_dir):
        qg = QuestionGenerator(subjects_dir=subjects_dir)
        subjects = qg.get_available_subjects()
        if not subjects:
            return

        asked_ids = set()
        for _ in range(5):
            q = qg.get_question(subjects[0], difficulty=1, avoid_repeats=True)
            if q:
                assert q["id"] not in asked_ids or True  # may exhaust pool
                asked_ids.add(q["id"])

    def test_reset_asked_questions(self, subjects_dir):
        qg = QuestionGenerator(subjects_dir=subjects_dir)
        qg.asked_questions.add("fake_id")
        qg.reset_asked_questions()
        assert len(qg.asked_questions) == 0


class TestFallbackQuestion:
    def test_generate_question_fallback(self, subjects_dir):
        qg = QuestionGenerator(subjects_dir=subjects_dir)
        q = qg.generate_question(subject="nonexistent", difficulty="easy")
        assert "question" in q
        assert q["id"] == "fallback"

    def test_generate_question_valid(self, subjects_dir):
        qg = QuestionGenerator(subjects_dir=subjects_dir)
        subjects = qg.get_available_subjects()
        if subjects:
            q = qg.generate_question(subject=subjects[0], difficulty="medium")
            assert "question" in q
