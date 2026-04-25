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
import time
import uuid
import logging
from collections import defaultdict, deque
from fastapi import Request
from fastapi.responses import JSONResponse
from config import settings


# ──────────────────────────────────────────────
#  Environment
# ──────────────────────────────────────────────

# Cleaned up server.py to use shared module


# ──────────────────────────────────────────────
#  FastAPI App Setup
# ──────────────────────────────────────────────

env_instance = AdaptiveTutorEnv()
logger = logging.getLogger("adaptive_tutor")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

REQUEST_LOG = deque(maxlen=1000)
RATE_BUCKET = defaultdict(deque)

def env_factory():
    """Factory function for OpenEnv."""
    return env_instance

app = create_fastapi_app(env_factory, TutorAction, TutorObservation)

@app.get("/health")
def health():
    return {"status": "ok", "service": "adaptive-tutor"}


@app.get("/health/liveness")
def liveness():
    return {"status": "alive"}


@app.get("/health/readiness")
def readiness():
    ready = True
    reasons = []
    if settings.lazy_load_ai:
        reasons.append("ai_model_lazy_load_enabled")
    if not os.path.exists("subjects"):
        ready = False
        reasons.append("subjects_dir_missing")
    return {"status": "ready" if ready else "not_ready", "reasons": reasons}

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
    profiles_dir = settings.profiles_dir
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
    start = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    path = request.url.path
    if settings.api_token and path.startswith("/metrics"):
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {settings.api_token}"
        if auth != expected:
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "request_id": request_id},
            )

    now = time.time()
    ip = request.client.host if request.client else "unknown"
    bucket = RATE_BUCKET[ip]
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= settings.rate_limit_per_minute:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limit_exceeded", "request_id": request_id},
        )
    bucket.append(now)

    if request.headers.get("x-forwarded-proto") == "https":
        request.scope["scheme"] = "https"
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("request_failed request_id=%s path=%s error=%s", request_id, path, str(exc))
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": request_id})

    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["x-request-id"] = request_id
    REQUEST_LOG.append(
        {
            "request_id": request_id,
            "path": path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
            "ts": int(now),
        }
    )
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s latency_ms=%s",
        request_id,
        request.method,
        path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/metrics/ops")
def ops_metrics():
    if not REQUEST_LOG:
        return {"requests": 0, "avg_latency_ms": 0.0, "error_rate": 0.0}
    total = len(REQUEST_LOG)
    avg_latency = round(sum(item["latency_ms"] for item in REQUEST_LOG) / total, 2)
    errors = sum(1 for item in REQUEST_LOG if item["status_code"] >= 500)
    return {
        "requests": total,
        "avg_latency_ms": avg_latency,
        "error_rate": round(errors / total, 4),
        "recent": list(REQUEST_LOG)[-100:],
    }


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
        host=settings.app_host,
        port=settings.app_port,
        forwarded_allow_ips="*",
        proxy_headers=True,
    )
