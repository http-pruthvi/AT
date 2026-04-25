"""
app.py — Interactive demo for AdaptiveTutor AI

This is the main Gradio app that lets you watch (or control) an AI tutor
teaching a simulated student in real-time. It connects to the OpenEnv
server running on port 7860 and optionally uses Ollama w/ Qwen2.5-3B
for smarter tutoring decisions.

If Ollama isn't running, it falls back to a simple heuristic agent —
still fun to watch, just not as clever.

Usage:
    python app.py
    # Opens on http://localhost:7861

Author: Pruthvi
Hackathon: Meta PyTorch OpenEnv Grand Finale, April 2026
"""

import json
import time
import random
import httpx
import gradio as gr
from typing import Dict, List, Any, Optional, Tuple

# --- config ---
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"
ENV_URL = "http://localhost:7860"  # the OpenEnv server

# keep track of the current episode
state = {
    "obs": None,
    "done": False,
    "step": 0,
    "total_reward": 0.0,
    "rewards": [],
    "breakdowns": [],
    "expert_events": [],
    "ollama_ok": False,
}


# ==========================================
#  LLM stuff (Ollama / Qwen2.5)
# ==========================================

def check_ollama():
    """Ping Ollama to see if it's up and has our model."""
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            # check if any model name contains "qwen2.5"
            found = any(OLLAMA_MODEL.split(":")[0] in m for m in models)
            state["ollama_ok"] = found
            return found
    except Exception:
        pass
    state["ollama_ok"] = False
    return False


def ask_ollama(prompt):
    """Send a prompt to Ollama and get the response."""
    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 256},
            },
            timeout=30.0,
        )
        if r.status_code == 200:
            return r.json().get("response", "")
    except Exception:
        pass
    return ""


def make_llm_prompt(obs):
    """Turn the current observation into a prompt the LLM can reason about."""
    profile = obs.get("student_profile", {})
    weak = ", ".join(profile.get("weak_concepts", [])[:5]) or "none identified"
    expert = obs.get("expert_feedback") or "No feedback yet"

    return f"""You are an adaptive AI tutor. Based on the student's current state, pick the best next action.

STUDENT STATE:
- Subject: {obs.get("current_topic", "unknown")}
- Current difficulty: {obs.get("current_difficulty", 1)}/5
- Last answer was: {"correct" if obs.get("student_correct") else "wrong"}
- Correct streak: {obs.get("consecutive_correct", 0)}
- Wrong streak: {obs.get("consecutive_wrong", 0)}
- Overall mastery: {profile.get("overall_mastery", 0):.1%}
- Weak areas: {weak}
- Expert says: {expert}

AVAILABLE ACTIONS: ask_question, give_hint, explain_concept, increase_difficulty, decrease_difficulty

Respond with ONLY a JSON object like this:
{{"action_type": "ask_question", "content": "optional explanation", "target_concept": "concept_name", "difficulty": 3}}"""


def parse_llm_response(raw, obs):
    """Try to extract a JSON action from LLM output. Fall back to heuristic if it's garbage."""
    try:
        # find the JSON blob in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end])
            if "action_type" in parsed:
                return {
                    "action_type": parsed.get("action_type", "ask_question"),
                    "content": parsed.get("content", ""),
                    "target_concept": parsed.get("target_concept", ""),
                    "difficulty": max(1, min(5, int(parsed.get("difficulty", 1)))),
                }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass  # LLM gave us junk, no worries

    # fall back to the simple heuristic
    return heuristic_action(obs)


def heuristic_action(obs):
    """
    Simple rule-based agent. Not fancy, but it works:
    - Struggling? Lower the difficulty or give a hint
    - On a streak? Crank it up
    - Otherwise just ask about the weakest concept
    """
    profile = obs.get("student_profile", {})
    weakest = profile.get("weakest_concept", "")
    wrong_streak = obs.get("consecutive_wrong", 0)
    correct_streak = obs.get("consecutive_correct", 0)
    diff = obs.get("current_difficulty", 1)

    if wrong_streak >= 2:
        if diff > 1:
            return {
                "action_type": "decrease_difficulty",
                "content": "Let's try something easier.",
                "target_concept": weakest,
                "difficulty": max(1, diff - 1),
            }
        return {
            "action_type": "give_hint",
            "content": "Here's a hint to help you out.",
            "target_concept": weakest,
            "difficulty": diff,
        }

    if correct_streak >= 3:
        return {
            "action_type": "increase_difficulty",
            "content": "Great work! Time for a challenge.",
            "target_concept": weakest,
            "difficulty": min(5, diff + 1),
        }

    # default: just ask a question about what they're worst at
    return {
        "action_type": "ask_question",
        "content": "",
        "target_concept": weakest,
        "difficulty": diff,
    }


# ==========================================
#  Environment interaction
# ==========================================

def env_reset():
    """Start a fresh episode."""
    try:
        r = httpx.post(f"{ENV_URL}/reset", timeout=10.0)
        data = r.json()
        obs = data.get("observation", data)

        # reset all our tracking
        state["obs"] = obs
        state["done"] = False
        state["step"] = 0
        state["total_reward"] = 0.0
        state["rewards"] = []
        state["breakdowns"] = []
        state["expert_events"] = []
        return obs
    except Exception as e:
        return {"error": str(e), "student_profile": {}, "current_topic": "N/A"}


def env_step(action):
    """Send an action to the environment and get back the result."""
    try:
        r = httpx.post(f"{ENV_URL}/step", json={"action": action}, timeout=10.0)
        data = r.json()
        obs = data.get("observation", data)
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        info = data.get("info", {})

        # update tracking
        state["obs"] = obs
        state["done"] = done
        state["step"] += 1
        state["total_reward"] += reward
        state["rewards"].append(reward)
        state["breakdowns"].append(info)

        # log expert stuff
        if obs.get("expert_preference_changed"):
            state["expert_events"].append(
                f"Step {state['step']}: Expert preferences SHIFTED!"
            )

        return obs, reward, done, info
    except Exception as e:
        return {"error": str(e)}, 0.0, True, {}


# ==========================================
#  UI rendering helpers
# ==========================================

def render_knowledge_map(obs):
    """Show what the student knows (and doesn't know) as a text bar chart."""
    profile = obs.get("student_profile", {})
    km = profile.get("knowledge_map", {})
    if not km:
        return "No knowledge data yet."

    # sort weakest first so it's obvious what needs work
    sorted_concepts = sorted(km.items(), key=lambda x: x[1])
    lines = []
    for concept, mastery in sorted_concepts:
        bar_len = int(mastery * 20)
        bar = "=" * bar_len + "." * (20 - bar_len)
        marker = "*" if mastery >= 0.8 else " "
        lines.append(f"{marker} {concept:25s} [{bar}] {mastery:.0%}")
    return "\n".join(lines)


def render_status(obs):
    """Build the top-of-page status cards as HTML."""
    profile = obs.get("student_profile", {})
    mastery = profile.get("overall_mastery", 0)
    topic = obs.get("current_topic", "N/A").upper()
    diff = obs.get("current_difficulty", 1)
    step = state["step"]
    total_r = state["total_reward"]

    # color the mastery bar based on progress
    if mastery >= 0.8:
        color = "#4ade80"  # green
    elif mastery >= 0.5:
        color = "#facc15"  # yellow
    else:
        color = "#f87171"  # red
    pct = int(mastery * 100)

    return f"""
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:16px;">
        <div style="background:linear-gradient(135deg,#667eea,#764ba2); padding:16px; border-radius:12px; text-align:center; color:white;">
            <div style="font-size:0.85em; opacity:0.8;">Subject</div>
            <div style="font-size:1.6em; font-weight:700;">{topic}</div>
        </div>
        <div style="background:linear-gradient(135deg,#f093fb,#f5576c); padding:16px; border-radius:12px; text-align:center; color:white;">
            <div style="font-size:0.85em; opacity:0.8;">Step / Difficulty</div>
            <div style="font-size:1.6em; font-weight:700;">{step}/20 &middot; Lvl {diff}</div>
        </div>
        <div style="background:linear-gradient(135deg,#4facfe,#00f2fe); padding:16px; border-radius:12px; text-align:center; color:white;">
            <div style="font-size:0.85em; opacity:0.8;">Total Reward</div>
            <div style="font-size:1.6em; font-weight:700;">{total_r:+.1f}</div>
        </div>
    </div>
    <div style="background:#1e1e2e; padding:12px 16px; border-radius:10px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="color:#cdd6f4;">Overall Mastery:</span>
            <div style="flex:1; background:#313244; border-radius:6px; height:20px; overflow:hidden;">
                <div style="width:{pct}%; background:{color}; height:100%; border-radius:6px; transition:width 0.3s;"></div>
            </div>
            <span style="color:{color}; font-weight:600;">{pct}%</span>
        </div>
    </div>
    """


def render_expert(obs):
    """Show expert feedback and any preference shift alerts."""
    fb = obs.get("expert_feedback")
    changed = obs.get("expert_preference_changed", False)
    events = state["expert_events"]

    if not fb and not events:
        return '<div style="color:#6c7086; font-style:italic;">No expert feedback yet. They chime in every few steps.</div>'

    html = ""
    if changed:
        html += """
        <div style="background:linear-gradient(135deg,#f5576c,#ff6a88); padding:12px 16px; border-radius:10px; color:white; margin-bottom:8px; font-weight:600;">
            Expert preferences have SHIFTED! The tutor needs to adapt.
        </div>"""

    if fb:
        html += f"""
        <div style="background:#1e1e2e; padding:12px 16px; border-radius:10px; border-left:4px solid #cba6f7;">
            <div style="color:#cba6f7; font-size:0.85em; margin-bottom:4px;">Expert says:</div>
            <div style="color:#cdd6f4; font-size:1.05em;">"{fb}"</div>
        </div>"""

    if events:
        html += '<div style="margin-top:8px; font-size:0.85em; color:#a6adc8;">'
        for e in events[-3:]:
            html += f"&bull; {e}<br>"
        html += "</div>"

    return html


def render_breakdown(info, reward):
    """Render the reward breakdown as a nice panel."""
    if not info:
        return ""

    components = [
        ("correctness", "Correctness", "#89b4fa"),
        ("mastery", "Mastery Bonus", "#a6e3a1"),
        ("difficulty", "Difficulty Match", "#f9e2af"),
        ("expert", "Expert Adapt", "#cba6f7"),
        ("efficiency", "Efficiency", "#fab387"),
    ]

    rows = ""
    for key, label, color in components:
        val = info.get(key, 0)
        sign = "+" if val >= 0 else ""
        val_color = "#a6e3a1" if val >= 0 else "#f38ba8"
        rows += f"""
        <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #313244;">
            <span style="color:{color};">{label}</span>
            <span style="color:{val_color}; font-weight:600;">{sign}{val:.1f}</span>
        </div>"""

    r_sign = "+" if reward >= 0 else ""
    r_color = "#a6e3a1" if reward >= 0 else "#f38ba8"

    return f"""
    <div style="background:#1e1e2e; padding:14px 16px; border-radius:10px;">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="color:#cdd6f4; font-weight:600;">Reward Breakdown</span>
            <span style="color:{r_color}; font-size:1.2em; font-weight:700;">{r_sign}{reward:.1f}</span>
        </div>
        {rows}
    </div>"""


# ==========================================
#  Main actions (button handlers)
# ==========================================

ALL_OUTPUTS = 9  # number of return values for our callbacks

def start_session():
    """Reset everything and start fresh."""
    obs = env_reset()
    has_ollama = check_ollama()

    if has_ollama:
        badge = '<span style="background:#a6e3a1;color:#1e1e2e;padding:3px 10px;border-radius:20px;font-size:0.85em;">Qwen2.5-3B via Ollama</span>'
    else:
        badge = '<span style="background:#f9e2af;color:#1e1e2e;padding:3px 10px;border-radius:20px;font-size:0.85em;">Heuristic Agent (no Ollama)</span>'

    return (
        render_status(obs),
        render_knowledge_map(obs),
        render_expert(obs),
        "",
        None,
        f"New session started! Using: {badge}",
        gr.update(interactive=True),   # step btn
        gr.update(interactive=True),   # simulate btn
        gr.update(interactive=True),   # full episode btn
    )


def do_step(use_llm=True):
    """Take one step in the episode."""
    obs = state["obs"]
    if obs is None or state["done"]:
        return _make_ui("Episode over. Hit 'New Session' to go again.")

    # pick an action (LLM or heuristic)
    if use_llm and state["ollama_ok"]:
        prompt = make_llm_prompt(obs)
        raw = ask_ollama(prompt)
        action = parse_llm_response(raw, obs)
        source = f"Qwen: {action['action_type']} -> {action.get('target_concept', '?')}"
    else:
        action = heuristic_action(obs)
        source = f"Heuristic: {action['action_type']} -> {action.get('target_concept', '?')}"

    # execute it
    obs, reward, done, info = env_step(action)

    # build the chart data
    import pandas as pd
    chart_data = []
    cumulative = 0
    for i, r in enumerate(state["rewards"], 1):
        cumulative += r
        chart_data.append({"Step": i, "Reward": round(r, 2), "Cumulative": round(cumulative, 2)})
    df = pd.DataFrame(chart_data) if chart_data else None

    # status message
    correct = "Correct" if obs.get("student_correct") else "Wrong"
    msg = f"Step {state['step']}: {source} | {correct} | Reward: {reward:+.1f}"

    if done:
        mastery = obs.get("student_profile", {}).get("overall_mastery", 0)
        if obs.get("mastery_achieved"):
            msg += f"\n\nMastery achieved in {state['step']} steps! Total: {state['total_reward']:.1f}"
        else:
            msg += f"\n\nEpisode done. Mastery: {mastery:.0%} | Total: {state['total_reward']:.1f}"

    return (
        render_status(obs),
        render_knowledge_map(obs),
        render_expert(obs),
        render_breakdown(info, reward),
        df,
        msg,
        gr.update(interactive=not done),
        gr.update(interactive=not done),
        gr.update(interactive=not done),
    )


def run_full_episode():
    """Auto-run all 20 steps. Yields results so the UI updates live."""
    yield start_session()
    time.sleep(0.5)

    for _ in range(20):
        if state["done"]:
            break
        yield do_step(use_llm=True)
        time.sleep(0.3)


def _make_ui(msg):
    """Helper to return current state with a message (e.g. when episode is done)."""
    obs = state["obs"] or {}
    return (
        render_status(obs),
        render_knowledge_map(obs),
        render_expert(obs),
        "", None, msg,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


# ==========================================
#  Build the Gradio UI
# ==========================================

with gr.Blocks(
    title="AdaptiveTutor AI",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue", neutral_hue="slate"),
    css=".gradio-container{max-width:1200px!important} footer{display:none!important}",
) as demo:

    gr.Markdown(
        "# AdaptiveTutor AI -- Live Demo\n"
        "**Meta PyTorch OpenEnv Hackathon** | "
        "Watch an AI tutor learn to teach through RL + shifting expert feedback"
    )

    with gr.Row():
        with gr.Column(scale=2):
            status_html = gr.HTML(value="<div style='color:#6c7086;'>Click 'New Session' to start.</div>")
            knowledge_box = gr.Code(label="Student Knowledge Map", language=None, interactive=False, lines=10)
            expert_html = gr.HTML(label="Expert Feedback")

        with gr.Column(scale=1):
            breakdown_html = gr.HTML(label="Reward Breakdown")
            chart = gr.Dataframe(label="Reward History", headers=["Step", "Reward", "Cumulative"],
                                 interactive=False, wrap=True)

    log = gr.Textbox(label="Action Log", interactive=False, lines=3)

    with gr.Row():
        btn_new = gr.Button("New Session", variant="primary", size="lg")
        btn_step = gr.Button("Next Step", variant="secondary", size="lg", interactive=False)
        btn_sim = gr.Button("Simulate Step", variant="secondary", size="lg", interactive=False)
        btn_full = gr.Button("Run Full Episode", variant="stop", size="lg", interactive=False)

    outputs = [status_html, knowledge_box, expert_html, breakdown_html, chart, log, btn_step, btn_sim, btn_full]

    btn_new.click(fn=start_session, outputs=outputs)
    btn_step.click(fn=lambda: do_step(use_llm=True), outputs=outputs)
    btn_sim.click(fn=lambda: do_step(use_llm=False), outputs=outputs)
    btn_full.click(fn=run_full_episode, outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
