# 🎓 AdaptiveTutor AI: Teaching LLMs to Teach

### *Recursive Self-Improvement for Personalized Education*

**By Pruthviraj Vinod Phuse**

---

## 🚀 The Vision: Why "Adaptive" Matters
In traditional education, the "one-size-fits-all" approach often leaves students behind or bores them to tears. LLMs have the potential to be the ultimate personal tutors, but they suffer from a major flaw: **they are static**. They don't learn from their mistakes or adjust their pedagogical style based on real-time student performance.

**AdaptiveTutor AI** changes that. Built for the **Meta PyTorch OpenEnv Hackathon**, this project introduces a teacher that learns to teach through **Reinforcement Learning from Human Feedback (RLHF)**—or in our case, **Reinforcement Learning from Expert Simulation (RLES)**.

---

## 🧠 The Architecture: How it Works
The project is built on three main pillars:

### 1. The OpenEnv Environment
Using the `openenv-core` framework, we've created a custom pedagogical environment. The "state" includes:
*   **Student Profile**: A dynamic model of the student's mastery in different concepts (e.g., Algebra, Geometry).
*   **Expert Feedback**: A simulated "Master Teacher" who provides periodic critiques of the tutor's actions.
*   **Session Progress**: Tracking how quickly a student is reaching the mastery threshold.

### 2. Multi-Factor Rewards
The AI doesn't just get a +1 for a correct answer. We calculate a high-dimensional reward based on:
*   **Learning Gain**: How much the student's mastery increased.
*   **Pedagogical Alignment**: Did the tutor follow the expert's advice?
*   **Efficiency**: Did they reach mastery in fewer steps?
*   **Concept Progression**: Are they moving logically from easy to hard topics?

### 3. Training with GRPO (Group Relative Policy Optimization)
We use the cutting-edge **GRPO** algorithm to train the model. Unlike standard PPO, GRPO compares a group of tutor responses for the same state, allowing the model to distinguish between "okay" teaching and "excellent" teaching.

---

## 🛠️ The Tech Stack
*   **Base Model**: `Qwen2.5-0.5B-Instruct` (Fast, efficient, and surprisingly smart).
*   **Training Framework**: `unsloth` & `trl` for ultra-fast fine-tuning in Colab.
*   **Backend**: `FastAPI` with an isolated API subpath for OpenEnv protocol compliance.
*   **UI/UX**: A premium `Gradio` dashboard with real-time visualization of the self-improvement loop.

---

## 📈 The Result: A Self-Improving Teacher
As the training loop progresses, the tutor evolves:
*   **Phase 1**: The tutor is generic, often asking too-easy or too-hard questions.
*   **Phase 2**: It begins to listen to the Expert Simulator, adjusting difficulty more accurately.
*   **Phase 3**: Mastery achieved. The tutor develops a "pedagogical intuition," recognizing exactly when a student is ready for a challenge and when they need a gentle hint.

---

## 🌟 Why This Wins
AdaptiveTutor AI isn't just an app; it's a **blueprint for recursive self-improvement**. By treating "teaching" as a quantifiable environment, we allow the AI to iterate on its own strategy, becoming more effective with every student it "teaches."

---

### *Check out the code and start the training loop on Hugging Face!*
[AdaptiveTutor AI on Hugging Face](https://huggingface.co/spaces/http-pruthvi/adaptive-tutor-ai)
