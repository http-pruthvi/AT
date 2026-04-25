"""
server.py - OpenEnv Environment Server for AdaptiveTutor AI

FastAPI server implementing the OpenEnv protocol (reset/step/state).
Orchestrates the student simulation, question generation, expert
feedback, and multi-factor reward calculation.

Episode flow:
1. reset() → initialize fresh student + load random subject
2. step(action) → evaluate tutor action → update state → return obs + reward
3. Episode ends when student reaches mastery (0.8+) OR 20 steps done
4. Expert feedback arrives randomly every 3-5 steps
"""

try:
    from openenv.core.env_server import create_fastapi_app
except (ImportError, ModuleNotFoundError):
    def create_fastapi_app(factory, action_model, obs_model):
        from fastapi import FastAPI
        return FastAPI()
from shared import AdaptiveTutorEnv, TutorAction, TutorObservation
import glob
import json
import os


# ──────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────

# Cleaned up server.py to use shared module


# ──────────────────────────────────────────────
#  FastAPI App Setup
# ──────────────────────────────────────────────

env_instance = AdaptiveTutorEnv()

def env_factory():
    """Factory function for OpenEnv."""
    return env_instance

app = create_fastapi_app(env_factory, TutorAction, TutorObservation)

@app.get("/health")
def health():
    return {"status": "ok", "service": "adaptive-tutor"}

@app.get("/metrics/session")
def session_metrics():
    student = env_instance.student
    if student is None:
        return {
            "step_count": env_instance.step_count,
            "episode_done": env_instance.episode_done,
            "overall_mastery": 0.0,
            "current_difficulty": env_instance.current_difficulty,
            "last_reward": env_instance.last_reward,
        }
    return {
        "step_count": env_instance.step_count,
        "episode_done": env_instance.episode_done,
        "overall_mastery": student.get_overall_mastery(),
        "current_difficulty": env_instance.current_difficulty,
        "last_reward": env_instance.last_reward,
        "last_reward_breakdown": env_instance.last_reward_breakdown,
        "weakest_concept": student.get_weakest_concept(),
    }

@app.get("/metrics/history")
def history_metrics():
    """Aggregate stored profile histories across students."""
    profiles_dir = "profiles"
    if not os.path.exists(profiles_dir):
        return {"students": 0, "sessions": 0, "avg_learning_gain_pct": 0.0, "records": []}

    records = []
    for path in glob.glob(os.path.join(profiles_dir, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for h in payload.get("history", []):
                records.append({
                    "student": payload.get("student_name", "unknown"),
                    "subject": h.get("subject"),
                    "timestamp": h.get("timestamp"),
                    "learning_gain_pct": h.get("metrics", {}).get("learning_gain_pct", 0.0),
                    "mastery_now_pct": h.get("metrics", {}).get("mastery_now_pct", 0),
                    "accuracy": h.get("accuracy", 0),
                })
        except Exception:
            continue

    if not records:
        return {"students": 0, "sessions": 0, "avg_learning_gain_pct": 0.0, "records": []}

    unique_students = len({r["student"] for r in records})
    avg_gain = round(sum(r["learning_gain_pct"] for r in records) / len(records), 2)
    return {
        "students": unique_students,
        "sessions": len(records),
        "avg_learning_gain_pct": avg_gain,
        "records": records[-300:],
    }

# --- HTTPS Proxy Support (HuggingFace Spaces) ---
# HF Spaces serves over HTTPS via a reverse proxy. Without this,
# Gradio generates http:// URLs causing "Mixed Content" browser errors.
@app.middleware("http")
async def force_https_scheme(request, call_next):
    """Trust X-Forwarded-Proto header from HF's reverse proxy."""
    if request.headers.get("x-forwarded-proto") == "https":
        request.scope["scheme"] = "https"
    return await call_next(request)


# ──────────────────────────────────────────────
#  Gradio Dashboard
# ──────────────────────────────────────────────
import gradio as gr
from app import demo

# Mount the rich interactive dashboard from app.py onto the root path
# so it becomes the main UI for HuggingFace Spaces.
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        forwarded_allow_ips="*",
        proxy_headers=True,
    )
