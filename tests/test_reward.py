"""Tests for all 5 independent reward functions in core/reward.py."""

from core.reward import RewardManager


class TestCorrectnessReward:
    def test_correct_answer(self):
        assert RewardManager.correctness_reward(True) == 1.0

    def test_wrong_answer(self):
        assert RewardManager.correctness_reward(False) == -0.3


class TestMasteryReward:
    def test_concept_at_mastery(self):
        km = {"algebra": 0.85}
        assert RewardManager.mastery_reward("algebra", km) == 2.0

    def test_concept_below_mastery(self):
        km = {"algebra": 0.5}
        assert RewardManager.mastery_reward("algebra", km) == 0.0

    def test_concept_exactly_at_threshold(self):
        km = {"algebra": 0.8}
        assert RewardManager.mastery_reward("algebra", km) == 2.0

    def test_missing_concept(self):
        km = {}
        assert RewardManager.mastery_reward("algebra", km) == 0.0


class TestDifficultyReward:
    def test_good_match(self):
        # Student at 0.5 mastery → expected difficulty ~3-4, chosen 3 → match
        r = RewardManager.difficulty_reward(
            chosen_difficulty=3, student_mastery=0.5,
            student_correct=True, consecutive_correct=0, consecutive_wrong=0,
        )
        assert r == 0.5

    def test_streak_not_increasing(self):
        # 3+ consecutive correct, but difficulty still low → penalty
        r = RewardManager.difficulty_reward(
            chosen_difficulty=2, student_mastery=0.4,
            student_correct=True, consecutive_correct=3, consecutive_wrong=0,
        )
        assert r == -0.5

    def test_struggling_not_decreasing(self):
        # 2+ consecutive wrong, but difficulty still high → penalty
        r = RewardManager.difficulty_reward(
            chosen_difficulty=4, student_mastery=0.6,
            student_correct=False, consecutive_correct=0, consecutive_wrong=2,
        )
        assert r == -0.5


class TestExpertReward:
    def test_adapted_to_feedback(self):
        r = RewardManager.expert_reward(
            expert_feedback="increase rigor", tutor_adapted=True,
        )
        assert r == 1.5

    def test_ignored_feedback(self):
        r = RewardManager.expert_reward(
            expert_feedback="increase rigor", tutor_adapted=False,
        )
        assert r == -1.0

    def test_no_feedback(self):
        r = RewardManager.expert_reward(expert_feedback=None, tutor_adapted=False)
        assert r == 0.0


class TestEfficiencyReward:
    def test_mastery_early(self):
        r = RewardManager.efficiency_reward(
            overall_mastery=0.85, steps_taken=10, max_steps=20,
        )
        # 3.0 * (20-10)/20 = 1.5
        assert r > 0.0
        assert r <= 3.0

    def test_no_mastery(self):
        r = RewardManager.efficiency_reward(
            overall_mastery=0.5, steps_taken=10, max_steps=20,
        )
        assert r == 0.0


class TestCalculateTotalReward:
    def test_reward_capping(self):
        """Total reward should never exceed MAX_REWARD_PER_STEP."""
        total, breakdown = RewardManager.calculate_total_reward(
            student_correct=True,
            concept="algebra",
            knowledge_map={"algebra": 0.9},
            chosen_difficulty=3,
            student_mastery=0.9,
            consecutive_correct=0,
            consecutive_wrong=0,
            expert_feedback="test",
            tutor_adapted=True,
            overall_mastery=0.9,
            steps_taken=5,
            max_steps=20,
        )
        assert total <= RewardManager.MAX_REWARD_PER_STEP

    def test_negative_capping(self):
        """Total reward should never go below MIN_REWARD_PER_STEP."""
        total, breakdown = RewardManager.calculate_total_reward(
            student_correct=False,
            concept="algebra",
            knowledge_map={"algebra": 0.1},
            chosen_difficulty=5,
            student_mastery=0.1,
            consecutive_correct=0,
            consecutive_wrong=3,
            expert_feedback="decrease difficulty",
            tutor_adapted=False,
            overall_mastery=0.1,
            steps_taken=19,
            max_steps=20,
        )
        assert total >= RewardManager.MIN_REWARD_PER_STEP

    def test_breakdown_has_all_components(self):
        _, breakdown = RewardManager.calculate_total_reward(
            student_correct=True,
            concept="algebra",
            knowledge_map={"algebra": 0.5},
            chosen_difficulty=3,
            student_mastery=0.5,
            consecutive_correct=0,
            consecutive_wrong=0,
            expert_feedback=None,
            tutor_adapted=False,
            overall_mastery=0.5,
            steps_taken=10,
            max_steps=20,
        )
        expected_keys = {"correctness", "mastery", "difficulty", "expert", "efficiency"}
        assert expected_keys.issubset(set(breakdown.keys()))
