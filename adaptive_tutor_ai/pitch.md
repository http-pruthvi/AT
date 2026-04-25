# AdaptiveTutor AI — 3-Minute Pitch

---

## Slide 1: The Hook

**"1 teacher. 40 students. Zero personalization."**

In India, 300 million students share classrooms with teacher ratios of 1:40.
Every student learns differently — but they all get the same lesson.

> *Open with this stat. Let it sit for 3 seconds.*

---

## Slide 2: The Problem — Static AI Tutors Don't Improve

Current AI tutoring apps are glorified flashcard machines:
- Fixed question sequences
- No adaptation to student struggles
- **They never learn to teach better**

It's like having a teacher who reads from a textbook — for every student, the same way, forever.

> *"What if the AI tutor could learn from every student interaction and improve itself?"*

---

## Slide 3: The Solution — AdaptiveTutor AI

An **OpenEnv reinforcement learning environment** where an AI tutor self-improves.

Three innovations:

1. **Self-Improving via RL** — GRPO training teaches the tutor which questions to ask,
   when to give hints, and how to adjust difficulty — all from reward signals.

2. **Realistic Student Model** — Probabilistic simulation of real learning: mastery grows,
   decays, and responds differently at each difficulty level.

3. **Shifting Expert Feedback** *(Snorkel AI Bonus)* — Subject experts change their
   teaching preferences every 5 steps, forcing the tutor to detect drift and adapt.

---

## Slide 4: Live Demo

**Show the Gradio app (app.py):**

1. Click "New Session" — see student knowledge map load
2. Click "Next Step" 3-4 times — watch rewards flow
3. Point out expert feedback appearing
4. Click "Run Full Episode" — watch 20 steps auto-run
5. Show reward chart climbing

> *"Notice how the tutor starts asking easy questions, then increases difficulty
> as the student improves. And when the expert feedback shifts — the tutor adapts."*

---

## Slide 5: Reward System — Multi-Dimensional, Anti-Gaming

5 independent reward functions (not one monolithic number):

| Function    | Value       | What it measures       |
| ----------- | ----------- | ---------------------- |
| Correctness | +1.0 / -0.3 | Did the student learn? |
| Mastery     | +2.0        | Concept hit 80%        |
| Difficulty  | +0.5 / -0.5 | Right challenge level? |
| Expert      | +1.5 / -1.0 | Adapted to feedback?   |
| Efficiency  | up to +3.0  | Fast mastery?          |

**Anti-hacking**: Total capped at 5.0/step. You can't game it by spamming easy questions.

> *"Each reward function targets a different quality of good teaching.
> The cap prevents degenerate strategies."*

---

## Slide 6: Snorkel AI Bonus — Shifting Expert Preferences

Three expert personas with rotating teaching philosophies:

- 🔢 **Dr. Sharma** (Math): Proof-based → Visual → Problem-solving → Proof-based
- 🔬 **Ms. Patel** (Science): Experimental → Conceptual → Inquiry → Experimental
- 📜 **Prof. Khan** (History): Sources → Narrative → Debate → Sources

**Every 5 steps**, preferences shift. The tutor must:
1. Detect the shift happened
2. Figure out the new preference
3. Adapt its strategy accordingly

This is the exact Snorkel challenge: **handling shifting annotation requirements**.

> *"It's not enough to learn one good strategy. The tutor must continuously adapt
> because the definition of 'good teaching' keeps changing."*

---

## Slide 7: Results & Training

| Metric            | Baseline   | After GRPO |
| ----------------- | ---------- | ---------- |
| Avg Reward        | ~14.6      | —          |
| Expert Adaptation | Random     | Learned    |
| Difficulty Match  | Rule-based | Adaptive   |

**Training pipeline:**
- Qwen2.5-3B as the base model
- GRPO via TRL library
- 100 environment trajectories → reward signals
- Colab notebook (T4 GPU)

> *Show the reward comparison chart from training.ipynb*

---

## Slide 8: Real World Impact

**300 million students** could benefit from AI tutors that actually improve.

The path to deployment:
1. **Today**: RL environment with simulated students ✓
2. **Next**: Replace simulated responses with real student data
3. **Scale**: Deploy as tutoring app, continuous fine-tuning from real interactions
4. **Impact**: Every student gets a tutor that learns their specific needs

This isn't just a hackathon project — it's the architecture for the future of personalized education.

---

## Q&A Prep

### "Why education?"
> India has 300 million students with a 1:40 teacher ratio. Personalized AI tutoring
> is the highest-leverage problem I can think of.

### "How is expert feedback different from normal RL?"
> In normal RL, the reward function is fixed. Here, expert preferences SHIFT every 5 steps.
> The tutor must detect drift and adapt — it can't just memorize one strategy.

### "Could this actually work in the real world?"
> Yes. The architecture is designed for it. Replace the simulated student with real
> responses, collect data, and fine-tune. The OpenEnv standard makes it easy to swap.

### "Why does this deserve the Snorkel AI bonus?"
> Snorkel's core challenge is handling shifting annotation preferences. Our expert
> simulator creates exactly this — changing requirements that the model must adapt to.
> It's not a bolt-on; it's central to the reward system.

### "Why multiple reward functions?"
> A single reward number hides what's going wrong. With 5 independent signals,
> we can diagnose *why* the tutor fails — bad difficulty choice? Ignoring expert feedback?
> Each one targets a specific tutoring quality.

---

## Timing Guide (3 minutes)

| Time | Slide    | Key Action                             |
| ---- | -------- | -------------------------------------- |
| 0:00 | Hook     | Drop the "1 teacher, 40 students" stat |
| 0:20 | Problem  | Explain why static tutors fail         |
| 0:40 | Solution | Three innovations, rapid-fire          |
| 1:00 | Demo     | Switch to live app, show 1 episode     |
| 1:40 | Rewards  | Explain multi-dimensional rewards      |
| 2:00 | Snorkel  | Expert preference shifting             |
| 2:20 | Results  | Show reward improvement chart          |
| 2:40 | Impact   | 300M students, path to deployment      |
| 3:00 | End      | "Questions?"                           |
