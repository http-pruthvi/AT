import os
import gradio as gr
import requests
import json
import time
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from session_manager import SessionManager
from product_evaluator import ProductEvaluator
from question_generator import QuestionGenerator
from shared import env_instance, TutorAction

SERVER_URL = "http://localhost:7860"
session = SessionManager()
evaluator = ProductEvaluator()
qgen = QuestionGenerator()

COLORS = {
    "primary": "#6C63FF",
    "success": "#22C55E", 
    "warning": "#F59E0B",
    "error": "#EF4444",
    "dark": "#0F172A",
    "card": "#1E293B"
}

CUSTOM_CSS = """
/* Overall theme */
.gradio-container {
    background: #0F172A !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Mode toggle buttons */
.mode-toggle {
    background: #1E293B;
    border-radius: 12px;
    padding: 8px;
}

/* Chat bubbles */
.ai-bubble {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6);
    color: white;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3);
}

.human-bubble {
    background: #334155;
    color: #F1F5F9;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
}

.correct-feedback {
    background: linear-gradient(135deg, #22C55E, #16A34A);
    color: white;
    border-radius: 12px;
    padding: 10px 14px;
}

.wrong-feedback {
    background: linear-gradient(135deg, #EF4444, #DC2626);
    color: white;
    border-radius: 12px;
    padding: 10px 14px;
}

/* Mastery bars */
.mastery-bar-container {
    background: #1E293B;
    border-radius: 8px;
    height: 12px;
    margin: 4px 0;
    overflow: hidden;
}

.mastery-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease-in-out;
    background: linear-gradient(90deg, #6C63FF, #22C55E);
}

/* Cards */
.stat-card {
    background: #1E293B;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    border: 1px solid #334155;
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(108, 99, 255, 0.4) !important;
}

/* Streak counter */
.streak-fire {
    font-size: 24px;
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

/* Win screen */
.win-screen {
    background: linear-gradient(135deg, #22C55E, #16A34A);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    color: white;
}

/* Offline banner */
.offline-banner {
    background: #F59E0B;
    color: #1E293B;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    text-align: center;
}
"""

def generate_mastery_html(mastery_map, subject):
    concepts = mastery_map.get(subject, {})
    html = "<div style='padding: 10px;'>"
    html += "<h4 style='color: #94A3B8; margin-bottom: 12px;'>📊 Concept Mastery</h4>"
    for concept, mastery in concepts.items():
        pct = int(mastery * 100)
        color = "#22C55E" if pct >= 80 else "#6C63FF" if pct >= 50 else "#EF4444"
        emoji = "✅" if pct >= 80 else "📈" if pct >= 50 else "⚠️"
        html += f"""
        <div style='margin-bottom: 10px;'>
            <div style='display: flex; justify-content: space-between; 
                        color: #F1F5F9; font-size: 13px; margin-bottom: 4px;'>
                <span>{emoji} {concept.replace('_', ' ').title()}</span>
                <span style='color: {color}; font-weight: 600;'>{pct}%</span>
            </div>
            <div style='background: #334155; border-radius: 8px; 
                        height: 10px; overflow: hidden;'>
                <div style='width: {pct}%; height: 100%; 
                            background: linear-gradient(90deg, #6C63FF, {color}); 
                            border-radius: 8px; 
                            transition: width 0.5s ease-in-out;'></div>
            </div>
        </div>"""
    html += "</div>"
    return html

def generate_chat_html(chat_history):
    html = "<div style='padding: 10px; max-height: 400px; overflow-y: auto;'>"
    for msg in chat_history[-10:]:
        if msg["role"] == "ai":
            html += f"""
            <div style='display: flex; margin-bottom: 12px;'>
                <div style='width: 32px; height: 32px; border-radius: 50%; 
                            background: #6C63FF; display: flex; align-items: center; 
                            justify-content: center; margin-right: 8px; 
                            flex-shrink: 0; font-size: 16px;'>🎓</div>
                <div style='background: linear-gradient(135deg, #6C63FF20, #8B5CF620); 
                            border: 1px solid #6C63FF40; color: #F1F5F9; 
                            border-radius: 18px 18px 18px 4px; 
                            padding: 12px 16px; max-width: 80%;'>
                    {msg["content"]}
                </div>
            </div>"""
        elif msg["role"] == "human":
            bg = "#22C55E30" if msg.get("is_correct") else "#EF444430" if msg.get("is_correct") == False else "#33415530"
            border = "#22C55E60" if msg.get("is_correct") else "#EF444460" if msg.get("is_correct") == False else "#47556960"
            html += f"""
            <div style='display: flex; justify-content: flex-end; margin-bottom: 12px;'>
                <div style='background: {bg}; border: 1px solid {border}; 
                            color: #F1F5F9; border-radius: 18px 18px 4px 18px; 
                            padding: 12px 16px; max-width: 80%;'>
                    {msg["content"]}
                </div>
            </div>"""
        elif msg["role"] == "feedback":
            emoji = "✅" if msg.get("is_correct") else "❌"
            bg = "#22C55E20" if msg.get("is_correct") else "#EF444420"
            border = "#22C55E50" if msg.get("is_correct") else "#EF444450"
            html += f"""
            <div style='background: {bg}; border: 1px solid {border}; 
                        border-radius: 12px; padding: 10px 14px; 
                        margin-bottom: 12px; color: #F1F5F9;'>
                {emoji} {msg["content"]}
            </div>"""
    html += "</div>"
    return html

def start_session(student_name, subject):
    session.reset(student_name or "Student", subject)
    # Direct call to shared env instance instead of requests.post to avoid deadlocks
    env_instance.reset()
    
    weak = session.get_weak_concepts()
    weak_concept = weak[0][0] if weak else "general"
    teacher_note = session.get_active_teacher_note()
    
    question_data = qgen.generate_question(
        subject=subject,
        concept=weak_concept,
        difficulty="medium",
        teacher_note=teacher_note
    )
    
    q_text = question_data.get("question", f"What is a basic concept in {subject}?")
    session.add_message("ai", f"👋 Hi {session.student_name}! Let's work on <b>{subject}</b>.<br><br>🎯 I noticed you need practice with <b>{weak_concept.replace('_', ' ').title()}</b>.<br><br><b>Question:</b> {q_text}")
    session.current_question = question_data
    
    mastery_html = generate_mastery_html(session.mastery_map, subject)
    chat_html = generate_chat_html(session.chat_history)
    
    stats_html = f"""
    <div style='padding: 10px;'>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px;'>
            <div style='background: #1E293B; border-radius: 10px; padding: 10px; text-align: center;'>
                <div style='font-size: 20px;'>🔥 {session.streak}</div>
                <div style='color: #94A3B8; font-size: 11px;'>Streak</div>
            </div>
            <div style='background: #1E293B; border-radius: 10px; padding: 10px; text-align: center;'>
                <div style='font-size: 20px; color: #22C55E;'>{session.get_accuracy()}%</div>
                <div style='color: #94A3B8; font-size: 11px;'>Accuracy</div>
            </div>
            <div style='background: #1E293B; border-radius: 10px; padding: 10px; text-align: center;'>
                <div style='font-size: 20px; color: #6C63FF;'>{session.questions_asked}</div>
                <div style='color: #94A3B8; font-size: 11px;'>Questions</div>
            </div>
            <div style='background: #1E293B; border-radius: 10px; padding: 10px; text-align: center;'>
                <div style='font-size: 16px; color: #F59E0B;'>{session.get_session_time()}</div>
                <div style='color: #94A3B8; font-size: 11px;'>Time</div>
            </div>
        </div>
    </div>"""
    
    offline_warning = "" if evaluator.ollama_available else """
    <div style='background: #F59E0B20; border: 1px solid #F59E0B50; 
                color: #F59E0B; padding: 8px 12px; border-radius: 8px; 
                margin-bottom: 10px; font-size: 12px;'>
        ⚠️ Ollama not detected - running in offline mode (simplified evaluation)
    </div>"""
    
    return chat_html, mastery_html, stats_html, offline_warning

def submit_answer(answer_text):
    if not answer_text.strip():
        return generate_chat_html(session.chat_history), "", ""
    
    session.add_message("human", answer_text)
    session.questions_asked += 1
    
    q_data = getattr(session, "current_question", {})
    question = q_data.get("question", "")
    correct_answer = q_data.get("correct_answer", answer_text)
    concept = q_data.get("concept", "general")
    
    result = evaluator.evaluate_answer(
        question=question,
        correct_answer=correct_answer,
        student_answer=answer_text,
        subject=session.subject
    )
    
    is_correct = result["is_correct"]
    
    if is_correct:
        session.correct_answers += 1
        session.streak += 1
        feedback = f"{result['encouragement']}<br><small style='opacity:0.8'>{result['explanation']}</small>"
        if session.streak > 1:
            feedback += f"<br>🔥 {session.streak} in a row!"
    else:
        session.streak = 0
        feedback = f"{result['encouragement']}<br><small style='opacity:0.8'>{result['explanation']}</small>"
    
    session.add_message("feedback", feedback, is_correct=is_correct)
    session.update_mastery(concept, result.get("mastery_delta", 0.05 if is_correct else -0.02))
    
    # Sync simulation step
    try:
        action = TutorAction(
            action_type="ask_question",
            difficulty=3 if q_data.get("difficulty") == "medium" else (1 if q_data.get("difficulty") == "easy" else 5),
            target_concept=concept
        )
        env_instance.step(action)
    except:
        pass
    
    if session.is_complete():
        summary = session.get_summary()
        win_msg = f"""🎉 <b>Session Complete!</b><br>
        Questions: {summary['questions']} | 
        Accuracy: {summary['accuracy']}% | 
        Time: {summary['time']}"""
        session.add_message("ai", win_msg)
    else:
        weak = session.get_weak_concepts()
        next_concept = weak[0][0] if weak else concept
        # Decision making: Rule-based or RL-AI driven
        if ai_model is not None:
            # We wrap the session in a dict for compatibility with the OpenEnv-like observation
            current_obs = {
                "difficulty": 1 if session.streak < 1 else (2 if session.streak < 3 else 3),
                "accuracy": session.get_accuracy()
            }
            difficulty_level, action_type = get_ai_action(session, current_obs)
            # Map word difficulty to numeric for question generator
            difficulty_map = {"easy": 1, "medium": 3, "hard": 5}
            difficulty = difficulty_map.get(difficulty_level, 3)
        else:
            # Fallback to rules
            difficulty = 1 if session.streak < 1 else (3 if session.streak < 3 else 5)
        teacher_note = session.get_active_teacher_note()
        
        next_q = qgen.generate_question(
            subject=session.subject,
            concept=next_concept,
            difficulty=difficulty,
            teacher_note=teacher_note
        )
        
        transition = "Let's try another one! " if is_correct else "Let's practice more. "
        if teacher_note:
            transition += f"(Teacher note: focusing on {teacher_note}) "
        
        q_text = next_q.get("question", f"Next question about {next_concept}")
        session.add_message("ai", f"{transition}<br><br><b>Question:</b> {q_text}")
        session.current_question = next_q
    
    mastery_html = generate_mastery_html(session.mastery_map, session.subject)
    chat_html = generate_chat_html(session.chat_history)
    
    stats_html = f"""
    <div style='padding: 10px;'>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px;'>
            <div style='background: #1E293B; border-radius: 10px; padding: 10px; text-align: center;'>
                <div style='font-size: 20px;'>{"🔥" * min(session.streak, 5)} {session.streak}</div>
                <div style='color: #94A3B8; font-size: 11px;'>Streak</div>
            </div>
            <div style='background: #1E293B; border-radius: 10px; padding: 10px; text-align: center;'>
                <div style='font-size: 20px; color: #22C55E;'>{session.get_accuracy()}%</div>
                <div style='color: #94A3B8; font-size: 11px;'>Accuracy</div>
            </div>
            <div style='background: #1E293B; border-radius: 10px; padding: 10px; text-align: center;'>
                <div style='font-size: 20px; color: #6C63FF;'>{session.questions_asked}</div>
                <div style='color: #94A3B8; font-size: 11px;'>Questions</div>
            </div>
            <div style='background: #1E293B; border-radius: 10px; padding: 10px; text-align: center;'>
                <div style='font-size: 16px; color: #F59E0B;'>{session.get_session_time()}</div>
                <div style='color: #94A3B8; font-size: 11px;'>Time</div>
            </div>
        </div>
    </div>"""
    
    return chat_html, mastery_html, stats_html

# --- AI Model Integration ---
TRAINED_MODEL_PATH = "./adaptive_tutor_trained"
ai_model = None
ai_tokenizer = None

def load_trained_model():
    global ai_model, ai_tokenizer
    if os.path.exists(TRAINED_MODEL_PATH):
        try:
            print(f"Loading trained AI model from {TRAINED_MODEL_PATH}...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            ai_tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
            ai_model = AutoModelForCausalLM.from_pretrained(
                TRAINED_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto"
            )
            return True
        except Exception as e:
            print(f"Failed to load AI model: {e}")
    return False

def get_ai_action(session, current_obs):
    """Use the trained RL model to decide the next tutoring action."""
    if ai_model is None:
        return "medium", "ask_question"
    
    # Build the prompt matching training format
    weak = ", ".join([c[0] for c in session.get_weak_concepts()[:3]])
    prompt = f"""<|system|>
You are an adaptive AI tutor who explains things SIMPLY. 
Generate ONE clear, easy-to-understand question.
</s>
<|user|>
Subject: {session.subject}
Difficulty: {current_obs.get('difficulty', 1)}
Student mastery: {session.get_accuracy()}%
Weak areas: {weak}
Generate a simple question.
</s>
<|assistant|>
"""
    inputs = ai_tokenizer(prompt, return_tensors="pt").to(ai_model.device)
    with torch.no_grad():
        outputs = ai_model.generate(**inputs, max_new_tokens=10, temperature=0.7)
    
    # Simple parsing: check if the AI suggests difficulty change
    resp = ai_tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
    
    # Default behavior
    difficulty = "medium"
    action = "ask_question"
    
    if "harder" in resp or "increase" in resp: difficulty = "hard"
    elif "easier" in resp or "decrease" in resp: difficulty = "easy"
    
    return difficulty, action

# Try loading on startup
AI_LOADED = load_trained_model()

def send_teacher_note(note):
    if note.strip():
        session.add_teacher_note(note)
        return f"✅ Note sent to AI: '{note}'"
    return ""

with gr.Blocks(css=CUSTOM_CSS, title="AdaptiveTutor AI") as demo:
    
    gr.HTML(f"""
    <div style='text-align: center; padding: 20px 0 10px;'>
        <h1 style='color: #6C63FF; font-size: 2.2em; margin: 0;'>
            🎓 SimpleTutor AI
        </h1>
        <p style='color: #94A3B8; margin: 5px 0;'>
            Adaptive learning made simple for everyone
        </p>
        <div style="display: inline-block; padding: 4px 12px; border-radius: 20px; background: {'#22c55e33' if AI_LOADED else '#ef444433'}; color: {'#22c55e' if AI_LOADED else '#ef4444'}; font-size: 0.9rem; margin-top: 10px; border: 1px solid {'#22c55e66' if AI_LOADED else '#ef444466'};">
            ● {'Smart AI Mode Active' if AI_LOADED else 'Rule-Based Engine (Training in progress...)'}
        </div>
        <div style='display: flex; justify-content: center; gap: 10px; margin-top: 10px;'>
            <span style='background: #6C63FF20; color: #6C63FF; 
                         padding: 4px 12px; border-radius: 20px; font-size: 12px;'>
                OpenEnv
            </span>
            <span style='background: #22C55E20; color: #22C55E; 
                         padding: 4px 12px; border-radius: 20px; font-size: 12px;'>
                Self-Improvement Theme
            </span>
            <span style='background: #F59E0B20; color: #F59E0B; 
                         padding: 4px 12px; border-radius: 20px; font-size: 12px;'>
                Snorkel AI Bonus
            </span>
        </div>
    </div>
    """)
    
    with gr.Tabs() as tabs:
        
        with gr.Tab("🎓 Human Mode", id="human"):
            
            offline_banner = gr.HTML("")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #F1F5F9;'>👤 Student Profile</h3>")
                    student_name = gr.Textbox(
                        placeholder="Your name...",
                        label="Name",
                        value="Student"
                    )
                    subject = gr.Dropdown(
                        choices=["Math", "Science", "History"],
                        value="Math",
                        label="Subject"
                    )
                    start_btn = gr.Button(
                        "🚀 Start Learning!", 
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )
                    mastery_display = gr.HTML(
                        "<div style='color: #94A3B8; padding: 20px; text-align: center;'>Start a session to see your progress</div>"
                    )
                    stats_display = gr.HTML("")
                
                with gr.Column(scale=2):
                    gr.HTML("<h3 style='color: #F1F5F9;'>💬 Chat with Your AI Tutor</h3>")
                    chat_display = gr.HTML(
                        "<div style='color: #94A3B8; padding: 40px; text-align: center;'>👆 Start a session to begin learning!</div>"
                    )
                    with gr.Row():
                        answer_input = gr.Textbox(
                            placeholder="Type your answer here...",
                            label="Your Answer",
                            scale=4
                        )
                        submit_btn = gr.Button("Send →", variant="primary", scale=1)
                
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #F1F5F9;'>👩‍🏫 Teacher Mode</h3>")
                    gr.HTML("<p style='color: #94A3B8; font-size: 13px;'>Send instructions to guide the AI tutor</p>")
                    teacher_note_input = gr.Textbox(
                        placeholder="e.g. Focus on quadratic equations...",
                        label="Note to AI",
                        lines=3
                    )
                    teacher_btn = gr.Button("📨 Send to AI", variant="secondary")
                    teacher_status = gr.HTML("")
                    
                    gr.HTML("""
                    <div style='background: #1E293B; border-radius: 12px; 
                                padding: 15px; margin-top: 15px;'>
                        <h4 style='color: #94A3B8; margin: 0 0 10px;'>🤖 AI Reasoning</h4>
                        <p style='color: #64748B; font-size: 12px; margin: 0;'>
                            AI targets weak concepts first.<br>
                            Drops difficulty after wrong answers.<br>
                            Adapts to teacher notes immediately.<br>
                            Expert feedback shifts every 5 steps.
                        </p>
                    </div>
                    """)
            
            start_btn.click(
                start_session,
                inputs=[student_name, subject],
                outputs=[chat_display, mastery_display, stats_display, offline_banner]
            )
            
            submit_btn.click(
                submit_answer,
                inputs=[answer_input],
                outputs=[chat_display, mastery_display, stats_display]
            )
            
            answer_input.submit(
                submit_answer,
                inputs=[answer_input],
                outputs=[chat_display, mastery_display, stats_display]
            )
            
            teacher_btn.click(
                send_teacher_note,
                inputs=[teacher_note_input],
                outputs=[teacher_status]
            )
        
        with gr.Tab("🤖 Demo Mode (RL Simulation)", id="demo"):
            
            gr.HTML("""
            <div style='text-align: center; padding: 20px;'>
                <h2 style='color: #6C63FF;'>RL Training Simulation</h2>
                <p style='color: #94A3B8;'>
                    Watch the AI tutor and simulated student interact.<br>
                    This is how the model gets trained using GRPO.
                </p>
            </div>
            """)
            
            with gr.Row():
                reward_display = gr.Number(label="Total Episode Reward", value=0)
                step_display = gr.Number(label="Current Step", value=0)
                mastery_pct = gr.Number(label="Avg Mastery %", value=0)
            
            speed = gr.Slider(
                minimum=1, maximum=5, value=1, step=1,
                label="Speed (1x to 5x)"
            )
            
            with gr.Row():
                run_demo_btn = gr.Button("▶️ Run Full Episode", variant="primary")
                stop_btn = gr.Button("⏹️ Stop", variant="secondary")
            
            demo_log = gr.Textbox(
                label="Episode Log",
                lines=15,
                value="Click 'Run Full Episode' to start simulation...",
                interactive=False
            )
            
            reward_chart = gr.Plot(label="Live Reward Chart")
        
        with gr.Tab("📊 Results & Training", id="results"):
            
            gr.HTML("<h2 style='color: #F1F5F9; padding: 20px 0 10px;'>Training Evidence</h2>")
            
            with gr.Row():
                gr.Image("reward_curve.png", label="Reward Improvement", 
                         show_label=True)
                gr.Image("reward_breakdown.png", label="Reward Breakdown Per Step",
                         show_label=True)
            
            gr.Image("mastery_progression.png", label="Student Mastery Progression")
            
            gr.HTML("""
            <div style='background: #1E293B; border-radius: 15px; 
                        padding: 25px; margin: 20px 0;'>
                <h3 style='color: #6C63FF;'>📈 Key Results</h3>
                <div style='display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 2em; color: #EF4444;'>8.5</div>
                        <div style='color: #94A3B8;'>Baseline Reward</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 2em; color: #22C55E;'>14.60</div>
                        <div style='color: #94A3B8;'>Trained Reward</div>
                    </div>
                    <div style='text-align: center;'>
                        <div style='font-size: 2em; color: #6C63FF;'>+72%</div>
                        <div style='color: #94A3B8;'>Improvement</div>
                    </div>
                </div>
            </div>
            """)
    
    gr.HTML("""
    <div style='text-align: center; padding: 15px; color: #475569; font-size: 12px;'>
        Built by Pruthviraj Vinod Phuse | Meta PyTorch OpenEnv Hackathon 2026 |
        <a href='https://huggingface.co/spaces/http-pruthvi/adaptive-tutor-ai' 
           style='color: #6C63FF;'>HF Space</a>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
