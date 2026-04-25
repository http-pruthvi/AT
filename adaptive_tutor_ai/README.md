---
title: AdaptiveTutor AI
emoji: "\U0001F393"
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: AI Tutor with Adaptive Expert Feedback (OpenEnv RL)
tags:
  - reinforcement-learning
  - education
  - openenv
  - pytorch
  - grpo
---

# AdaptiveTutor AI

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blueviolet?style=for-the-badge)](https://github.com/pytorch/openenv)
[![PyTorch](https://img.shields.io/badge/PyTorch-Hackathon-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![TRL](https://img.shields.io/badge/TRL-GRPO-green?style=for-the-badge)](https://huggingface.co/docs/trl)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-orange?style=for-the-badge&logo=gradio)](https://gradio.app)

**Meta PyTorch OpenEnv Hackathon Grand Finale — April 2026**

---

## The problem nobody talks about

In India, the average classroom has **1 teacher for 40 students**. That's 300 million kids who never get personalized attention. Every student is different — different strengths, different gaps, different pace — but they all get the exact same lesson.

AI tutoring apps were supposed to fix this. They didn't. Most of them are just flashcard apps with a chat interface. They follow fixed scripts, they don't adapt when a student is struggling, and worst of all — **they never get better at teaching**.

## What this project actually does

AdaptiveTutor AI is an **OpenEnv reinforcement learning environment** where an AI tutor genuinely learns to teach better over time. Three things make it different:

**1. It improves through RL.** The tutor starts out pretty bad. Through GRPO training, it learns which questions to ask, when to back off, and how to match difficulty to the student's level. All from reward signals, no hand-coded rules.

**2. The student is realistic.** Not a random number generator — a probabilistic model where mastery grows with correct answers, decays over time, and responds differently depending on difficulty level. It actually behaves like a student.

**3. The experts keep changing their minds.** This is the Snorkel AI bonus. Three subject matter experts periodically shift their teaching preferences. Dr. Sharma starts wanting proof-based math, then switches to visual intuition. The tutor has to notice the shift and adapt. You can't just memorize one good strategy — the definition of "good teaching" keeps moving.

---

## How it works

```
                    AdaptiveTutor AI
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  server.py (OpenEnv)                         │
  │  ├── student_model.py   (simulated learner)  │
  │  ├── question_generator (75 real questions)   │
  │  ├── expert_simulator   (Snorkel AI bonus)   │
  │  └── reward.py          (5 reward functions)  │
  │                                              │
  │  app.py ──── Gradio demo (Ollama + fallback) │
  │  client.py ── heuristic demo agent           │
  │  training.ipynb ── GRPO training on Colab    │
  │                                              │
  └──────────────────────────────────────────────┘
```

## The expert shift thing (Snorkel AI bonus)

This is the part I'm most proud of. Every 5 steps, the expert's teaching philosophy rotates:

| Expert | Subject | They start wanting... | Then switch to... | Then to... |
|--------|---------|----------------------|-------------------|------------|
| Dr. Sharma | Math | Rigorous proofs | Visual intuition | Problem-solving |
| Ms. Patel | Science | Hands-on experiments | Conceptual understanding | Student-led inquiry |
| Prof. Khan | History | Primary sources | Narrative storytelling | Structured debate |

The tutor gets +1.5 reward for adapting to these shifts and -1.0 for ignoring them. This creates a non-stationary reward landscape — exactly the kind of challenge Snorkel AI's theme is about.

## Reward system

I split the reward into 5 independent functions instead of one big number. Each one measures something different about tutoring quality:

| What it measures | Good | Bad | Why it matters |
|-----------------|------|-----|----------------|
| Did the student get it right? | +1.0 | -0.3 | Basic: are we teaching effectively? |
| Did a concept hit 80% mastery? | +2.0 | — | Milestone bonus for real progress |
| Was the difficulty appropriate? | +0.5 | -0.5 | Too hard = frustration, too easy = boredom |
| Did we listen to the expert? | +1.5 | -1.0 | Adaptability to shifting requirements |
| How fast did we reach mastery? | up to +3.0 | — | Efficiency matters in real classrooms |

**Anti-gaming cap**: Total reward is clamped at 5.0 per step. You can't hack the reward by spamming easy questions.

## Quick start

```bash
# run the environment
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860

# in another terminal, run the demo
python app.py

# or just run the heuristic agent
python client.py
```

Docker:
```bash
docker build -t adaptive-tutor-ai .
docker run -p 7860:7860 adaptive-tutor-ai
```

Training: upload `training.ipynb` to Colab with a T4 GPU.

## Files

| File | What it does |
|------|-------------|
| `server.py` | OpenEnv environment — reset/step/state endpoints |
| `app.py` | Gradio demo with Ollama integration |
| `client.py` | Heuristic agent + Pydantic models |
| `student_model.py` | Probabilistic student simulation |
| `question_generator.py` | Loads and serves questions from JSON banks |
| `expert_simulator.py` | The shifting expert feedback system |
| `reward.py` | All 5 reward functions + anti-hacking |
| `training.ipynb` | GRPO training notebook for Colab |
| `pitch.md` | 3-minute pitch script + slide outline |
| `subjects/*.json` | 75 questions across math, science, history |

## FAQ (for judges)

**Why education?**
300 million students in India, 1:40 teacher ratio. This is the highest-leverage problem I could think of.

**How is expert feedback different from normal RL?**
In normal RL the reward function doesn't change. Here, expert preferences shift every 5 steps. The tutor can't just learn one strategy — it has to detect drift and adapt on the fly.

**Could this actually work in the real world?**
The architecture is designed for it. Replace the simulated student with real answers, collect data, keep training. The OpenEnv standard makes it modular.

**Why does this deserve the Snorkel bonus?**
Snorkel's whole thing is handling shifting annotation/labeling requirements. That's exactly what the expert simulator does — the "correct" teaching approach keeps changing, and the model has to keep up.

---

MIT License — Built for the Meta PyTorch OpenEnv Hackathon 2026.
