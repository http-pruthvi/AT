---
title: AdaptiveTutor AI
emoji: 🎓
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: An AI tutor that learns and adapts in real-time.
---

# 🎓 AdaptiveTutor AI

AdaptiveTutor AI is a next-generation pedagogical platform that uses **Reinforcement Learning (RL)** and **Group Relative Policy Optimization (GRPO)** to provide personalized, adaptive learning experiences.

## 🚀 Features

- **Smart AI Mode**: Powered by a local Qwen-0.5B-Instruct model (quantized for efficiency).
- **Real-time Student Modeling**: Adapts difficulty and topics based on student performance.
- **Teacher Integration**: Allows teachers to send real-time guidance notes to the AI tutor.
- **Self-Improvement Loop**: Learns from interaction logs to optimize its teaching policy.
- **Stunning UI**: Premium Gradio interface with glassmorphism and live mastery tracking.

## 🛠️ Technology Stack

- **Core**: Python 3.11+
- **AI/ML**: PyTorch, Transformers, BitsAndBytes, TRL (GRPO)
- **UI**: Gradio
- **RL Environment**: Custom OpenEnv-compliant tutor environment

## 📖 How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. (Optional) Run Ollama for enhanced reasoning:
   ```bash
   ollama run qwen2.5:0.5b
   ```

## 🧪 Research & Development

For a deep dive into the underlying technology and research, check out:
- [HOW_IT_WORKS_SIMPLE.md](HOW_IT_WORKS_SIMPLE.md) (Child-friendly explanation)
- [BLOG_POST.md](BLOG_POST.md) (Full technical breakdown)
