# AdaptiveTutor AI — Pitch (3 min)

---

## Slide 1 — Hook

"1 teacher. 40 students. Zero personalization."

In India alone, 300 million students sit in classrooms where no one has time to figure out what they individually need. Every kid is different, but they all get the same lesson, the same pace, the same questions.

(Pause. Let the number sink in.)

---

## Slide 2 — Why current AI tutors fail

Most AI tutoring apps are basically flashcard machines with a chat bubble.
They follow fixed scripts. They don't notice when a student is struggling.
And here's the real problem: they never get better at teaching.

"What if the tutor itself could learn from every interaction — and actually improve?"

---

## Slide 3 — What we built

AdaptiveTutor AI is an OpenEnv RL environment. The AI tutor learns to teach through three mechanisms:

1. **Self-improvement via GRPO** — it starts bad, gets better through training
2. **Realistic student sim** — not random numbers, actual learning dynamics
3. **Shifting expert feedback** — this is the Snorkel AI bonus. Three experts rotate their teaching preferences every 5 steps. The tutor has to detect the shift and adapt.

---

## Slide 4 — Live demo

(Switch to the Gradio app)

- Hit "New Session" — watch the knowledge map load
- Click through 3-4 steps — see rewards flowing, student responding
- Point out when expert feedback appears
- Hit "Run Full Episode" — watch 20 steps auto-play
- Show the reward accumulating

"See how it starts with easy questions, then ramps up? And when the expert shifts preferences at step 5, the tutor adjusts. That adaptation is what we're training."

---

## Slide 5 — Reward system

We don't use one big reward number. Five independent signals:

- Correctness (+1.0 / -0.3) — is the student learning?
- Mastery (+2.0) — did a concept cross 80%?
- Difficulty match (+/-0.5) — right challenge level?
- Expert alignment (+1.5 / -1.0) — are we listening to the expert?
- Efficiency (up to +3.0) — did we get there fast?

Total capped at 5.0 per step. You can't game it by asking easy questions over and over.

"Each function catches a different failure mode. The cap prevents reward hacking."

---

## Slide 6 — Snorkel AI bonus

Three experts, each with rotating philosophies:

- Dr. Sharma (Math): proofs → visual → problem-solving → proofs again
- Ms. Patel (Science): experiments → concepts → inquiry → experiments
- Prof. Khan (History): primary sources → narrative → debate → sources

Every 5 steps, preferences rotate. The tutor must:
1. Notice something changed
2. Figure out what the expert wants now
3. Change strategy

This is exactly Snorkel's challenge: "How do you handle shifting annotation requirements?"
We didn't bolt it on. It's central to the reward.

---

## Slide 7 — Results

| | Baseline | After GRPO |
|---|---------|-----------|
| Avg episode reward | ~14.6 | TBD |
| Expert adaptation | Random | Learned |
| Difficulty matching | Rule-based | Adaptive |

(Show the reward comparison chart from training.ipynb)

---

## Slide 8 — Why this matters

This isn't just a hackathon project. The architecture is designed for real deployment:

1. Today: RL environment with simulated students (done)
2. Next: swap in real student responses
3. Scale: continuous fine-tuning from actual interactions
4. Impact: every student gets a tutor that learns *their* needs

300 million students. One tutor that gets better with each one.

"Questions?"

---

## Q&A cheat sheet

**"Why education?"**
India has 300M students with 1:40 teacher ratios. This is the highest-leverage application I could think of.

**"Expert feedback vs normal RL?"**
Normal RL has a fixed reward function. Ours shifts every 5 steps. The tutor can't memorize one strategy — it must detect drift and adapt.

**"Real world deployment?"**
Replace simulated student with real responses. The OpenEnv interface makes it modular — same environment, real data.

**"Why Snorkel bonus?"**
Snorkel's core challenge is shifting labeling preferences. Our expert simulator is literally that — changing definitions of "good teaching" that the model must track.

**"Why multiple reward functions?"**
One number hides what's going wrong. With five signals, I can diagnose whether the problem is difficulty selection, expert adaptation, or something else.

---

## Timing

| Time | What to say |
|------|------------|
| 0:00-0:20 | Hook + the stat |
| 0:20-0:40 | Why current tutors fail |
| 0:40-1:00 | Our three innovations |
| 1:00-1:40 | Live demo (biggest chunk) |
| 1:40-2:00 | Reward system |
| 2:00-2:20 | Snorkel bonus |
| 2:20-2:40 | Results chart |
| 2:40-3:00 | Impact + close |
