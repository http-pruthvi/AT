"""Tests for the AdaptiveTutorEnv — full episode lifecycle."""

from shared import AdaptiveTutorEnv, TutorAction


class TestEnvironmentReset:
    def test_reset_returns_observation(self):
        env = AdaptiveTutorEnv()
        obs = env.reset()
        assert obs.current_topic != ""
        assert obs.done is False
        assert obs.session_progress == 0.0
        assert obs.current_difficulty == 1

    def test_reset_initializes_student(self):
        env = AdaptiveTutorEnv()
        env.reset()
        assert env.student is not None
        assert env.step_count == 0
        assert env.episode_done is False


class TestEnvironmentStep:
    def test_single_step(self):
        env = AdaptiveTutorEnv()
        env.reset()
        action = TutorAction(
            action_type="ask_question",
            difficulty=1,
            target_concept="",
        )
        obs = env.step(action)
        assert env.step_count == 1
        assert obs.session_progress > 0.0

    def test_reward_is_numeric(self):
        env = AdaptiveTutorEnv()
        env.reset()
        action = TutorAction(
            action_type="ask_question",
            difficulty=2,
            target_concept="",
        )
        obs = env.step(action)
        assert isinstance(obs.reward, float)

    def test_difficulty_increase(self):
        env = AdaptiveTutorEnv()
        env.reset()
        action = TutorAction(
            action_type="increase_difficulty",
            difficulty=1,
            target_concept="",
        )
        obs = env.step(action)
        assert env.current_difficulty == 2

    def test_difficulty_decrease_floors_at_1(self):
        env = AdaptiveTutorEnv()
        env.reset()
        action = TutorAction(
            action_type="decrease_difficulty",
            difficulty=1,
            target_concept="",
        )
        obs = env.step(action)
        assert env.current_difficulty >= 1


class TestFullEpisode:
    def test_episode_terminates_at_max_steps(self):
        env = AdaptiveTutorEnv()
        env.reset()
        for _ in range(env.MAX_STEPS):
            action = TutorAction(
                action_type="ask_question",
                difficulty=3,
                target_concept="",
            )
            obs = env.step(action)
        assert obs.done is True
        assert env.step_count == env.MAX_STEPS

    def test_episode_accumulates_reward(self):
        env = AdaptiveTutorEnv()
        env.reset()
        total = 0.0
        for _ in range(5):
            action = TutorAction(
                action_type="ask_question",
                difficulty=2,
                target_concept="",
            )
            obs = env.step(action)
            total += obs.reward
        # Reward can be positive or negative, but it should be a number
        assert isinstance(total, float)

    def test_state_matches_last_step(self):
        env = AdaptiveTutorEnv()
        env.reset()
        action = TutorAction(
            action_type="ask_question",
            difficulty=1,
            target_concept="",
        )
        step_obs = env.step(action)
        state_obs = env.state()
        assert step_obs.step_count == state_obs.step_count if hasattr(step_obs, 'step_count') else True
        assert state_obs.current_difficulty == env.current_difficulty
