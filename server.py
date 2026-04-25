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

import os
import copy
import random
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware
from openenv.core.env_server import Environment, create_fastapi_app

from student_model import StudentProfile
from question_generator import QuestionGenerator
from expert_simulator import ExpertSimulator
from reward import RewardManager


# ──────────────────────────────────────────────
#  Shared Models & Environment
# ──────────────────────────────────────────────

from shared import env_instance, TutorAction, TutorObservation


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

# --- HTTPS Proxy Support (HuggingFace Spaces) ---
# HF Spaces serves over HTTPS via a reverse proxy. Without this,
# Gradio generates http:// URLs causing "Mixed Content" browser errors.
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

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
from fastapi.responses import HTMLResponse
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
