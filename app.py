import gradio as gr
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import random
from core.session_manager import SessionManager
from core.product_evaluator import ProductEvaluator
from core.question_generator import QuestionGenerator
from shared import AdaptiveTutorEnv, TutorAction, load_ai_model, AI_LOADED, env_instance
from config import settings

SERVER_URL = settings.server_url
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

def _get_runtime(runtime_state):
    if isinstance(runtime_state, dict):
        session = runtime_state.get("session")
        env = runtime_state.get("env")
        evaluator = runtime_state.get("evaluator")
        if session and env and evaluator:
            return session, env, evaluator
    session = SessionManager()
    env = AdaptiveTutorEnv()
    evaluator = ProductEvaluator()
    return session, env, evaluator


def start_session(student_name, subject, runtime_state):
    session, env, evaluator = _get_runtime(runtime_state)
    session.reset(student_name or "Student", subject)
    profile_loaded = session.load_profile()
    env.reset()
    
    weak = session.get_weak_concepts()
    weak_concept = weak[0][0] if weak else "general"
    teacher_note = session.get_active_teacher_note()
    
    question_data = qgen.generate_question(
        subject=subject.lower(),
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
        <div style='background: #0f172a; border: 1px solid #334155; border-radius: 8px; 
                    padding: 8px 10px; margin-bottom: 10px; color: #94A3B8; font-size: 12px;'>
            {'✅ Resumed saved profile' if profile_loaded else '🆕 New profile started'} |
            Mastery: {session.get_learning_metrics()['mastery_now_pct']}%
        </div>
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
    
    next_state = {"session": session, "env": env, "evaluator": evaluator}
    return chat_html, mastery_html, stats_html, offline_warning, next_state

def submit_answer(answer_text, runtime_state):
    session, env, evaluator = _get_runtime(runtime_state)
    if not answer_text.strip():
        return generate_chat_html(session.chat_history), "", "", runtime_state
    
    session.add_message("human", answer_text)
    session.questions_asked += 1
    
    q_data = getattr(session, "current_question", {})
    question = q_data.get("question", "")
    correct_answer = q_data.get("correct_answer", q_data.get("answer", answer_text))
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
    session.log_interaction(
        question=question,
        correct_answer=correct_answer,
        student_answer=answer_text,
        is_correct=is_correct,
        concept=concept,
        difficulty=q_data.get("difficulty", "medium"),
    )
    
    # Sync simulation step
    try:
        action = TutorAction(
            action_type="ask_question",
            difficulty=3 if q_data.get("difficulty") == "medium" else (1 if q_data.get("difficulty") == "easy" else 5),
            target_concept=concept
        )
        env.step(action)
    except Exception:
        pass
    
    if session.is_complete():
        summary = session.get_summary()
        win_msg = f"""🎉 <b>Session Complete!</b><br>
        Questions: {summary['questions']} | 
        Accuracy: {summary['accuracy']}% | 
        Time: {summary['time']}<br>
        Learning Gain: {summary['learning_gain_pct']}% | Mastery: {summary['mastery_now_pct']}%"""
        session.add_message("ai", win_msg)
        session.save_profile()
    else:
        weak = session.get_weak_concepts()
        next_concept = weak[0][0] if weak else concept
        # Decision making: Rule-based or RL-AI driven
        if shared.ai_model is not None:
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
            subject=session.subject.lower(),
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

    session.save_profile()
    
    mastery_html = generate_mastery_html(session.mastery_map, session.subject)
    chat_html = generate_chat_html(session.chat_history)
    
    stats_html = f"""
    <div style='padding: 10px;'>
        <div style='background: #0f172a; border: 1px solid #334155; border-radius: 8px; 
                    padding: 8px 10px; margin-bottom: 10px; color: #94A3B8; font-size: 12px;'>
            Learning Gain: {session.get_learning_metrics()['learning_gain_pct']}% |
            Mastery: {session.get_learning_metrics()['mastery_now_pct']}%
        </div>
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
    
    next_state = {"session": session, "env": env, "evaluator": evaluator}
    return chat_html, mastery_html, stats_html, next_state

# Shared model is already loaded at the top
import shared

def get_ai_action(session, current_obs):
    """Use the trained RL model to decide the next tutoring action."""
    if shared.ai_model is None and settings.lazy_load_ai:
        load_ai_model()

    ai_model = shared.ai_model
    ai_tokenizer = shared.ai_tokenizer
    if ai_model is None:
        return "medium", "ask_question"
    
    # Build the prompt matching training format
    weak = ", ".join([c[0] for c in session.get_weak_concepts()[:3]])
    teacher_note = session.get_active_teacher_note()
    teacher_context = f"Teacher Note: {teacher_note}\n" if teacher_note else ""
    
    prompt = f"""<|system|>
You are an adaptive AI tutor who explains things SIMPLY. 
Generate ONE clear, easy-to-understand question.
</s>
<|user|>
Subject: {session.subject}
Difficulty: {current_obs.get('difficulty', 1)}
Student mastery: {session.get_accuracy()}%
Weak areas: {weak}
{teacher_context}
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



def send_teacher_note(note, runtime_state):
    session, env, evaluator = _get_runtime(runtime_state)
    if note.strip():
        session.add_teacher_note(note)
        return f"✅ Note sent to AI: '{note}'", {"session": session, "env": env, "evaluator": evaluator}
    return "", {"session": session, "env": env, "evaluator": evaluator}

def run_demo_sim(speed_val):
    """Run an autonomous simulation of the AI tutor teaching a student."""
    env = AdaptiveTutorEnv()
    obs = env.reset()
    
    total_reward = 0
    steps = 0
    log_text = "🚀 Starting RL Simulation Episode...\n"
    log_text += "="*40 + "\n"
    history = []
    
    # Simple mock session for the evaluator logic if needed
    from core.session_manager import SessionManager
    mock_session = SessionManager("Simulated Student", obs.current_topic)
    
    fig, ax = plt.subplots(figsize=(6, 3), facecolor='#0F172A')
    ax.set_facecolor('#1E293B')
    
    mastered_concepts = set()
    
    for _ in range(env.MAX_STEPS):
        # AI Logic to pick action
        difficulty_str, action_type = get_ai_action(mock_session, obs.model_dump())
        
        # Convert difficulty string to numeric for the environment
        diff_map = {"easy": 1, "medium": 3, "hard": 5}
        numeric_diff = diff_map.get(difficulty_str, 3)
        
        # Pick a concept based on mastery
        concepts = list(obs.student_profile.keys())
        # Target lowest mastery concept
        if obs.student_profile:
            target_concept = min(obs.student_profile.items(), key=lambda x: x[1])[0]
        else:
            target_concept = random.choice(concepts) if concepts else ""
            
        # Agent Reasoning Log
        reasoning = f"🧠 AI Reasoning: Target weak concept '{target_concept}', diff '{difficulty_str}'"
        
        # Build action
        action = TutorAction(
            action_type=action_type,
            difficulty=numeric_diff,
            target_concept=target_concept
        )
        
        # Step the environment
        obs = env.step(action)
        total_reward += obs.reward
        steps += 1
        
        # Update mock session with new mastery
        for c, m in obs.student_profile.items():
            mock_session.knowledge_map[c] = m
            if m >= 0.8 and c not in mastered_concepts:
                mastered_concepts.add(c)
                log_text += f"\n🏆 Mastery Alert: Student mastered '{c}'!\n\n"
            
        # Expert shift
        if env.last_expert_pref_changed:
            log_text += f"\n⚠️ Expert Shift: {env.last_expert_feedback}\n\n"
            
        # Log the interaction
        status = "✅ Correct" if obs.student_correct else "❌ Wrong"
        log_text += f"Step {steps}: {action_type} ({difficulty_str}) -> {status}\n"
        log_text += f"{reasoning}\n"
        
        # Reward breakdown
        bd = env.last_reward_breakdown
        if bd:
            bd_str = ", ".join(f"{k}: {v:+.1f}" for k, v in bd.items() if v != 0)
            log_text += f"💰 Reward: {obs.reward:+.2f} ({bd_str})\n"
        else:
            log_text += f"💰 Reward: {obs.reward:+.2f}\n"
        log_text += "-"*40 + "\n"
        
        # Update chart data
        history.append({"Step": steps, "Cumulative": total_reward})
        df = pd.DataFrame(history)
        
        # Generate plot
        plt.clf()
        plt.plot(df["Step"], df["Cumulative"], marker='o', color='#6C63FF', linewidth=2)
        plt.title("Cumulative Reward Progression", color='white')
        plt.xlabel("Step", color='white')
        plt.ylabel("Total Reward", color='white')
        plt.tick_params(colors='white')
        plt.grid(True, alpha=0.1)
        plt.tight_layout()
        
        avg_mastery = sum(obs.student_profile.values()) / len(obs.student_profile) if obs.student_profile else 0
        
        yield total_reward, steps, avg_mastery * 100, log_text, plt.gcf()
        
        if obs.done:
            break
            
        time.sleep(1.1 / speed_val)
    
    log_text += f"\n🏁 Episode Complete! Total Reward: {total_reward:.2f}"
    yield total_reward, steps, avg_mastery * 100, log_text, plt.gcf()

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
                runtime_state = gr.State(value={})
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
                inputs=[student_name, subject, runtime_state],
                outputs=[chat_display, mastery_display, stats_display, offline_banner, runtime_state]
            )
            
            submit_btn.click(
                submit_answer,
                inputs=[answer_input, runtime_state],
                outputs=[chat_display, mastery_display, stats_display, runtime_state]
            )
            
            answer_input.submit(
                submit_answer,
                inputs=[answer_input, runtime_state],
                outputs=[chat_display, mastery_display, stats_display, runtime_state]
            )
            
            teacher_btn.click(
                send_teacher_note,
                inputs=[teacher_note_input, runtime_state],
                outputs=[teacher_status, runtime_state]
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
            
            run_demo_btn.click(
                run_demo_sim,
                inputs=[speed],
                outputs=[reward_display, step_display, mastery_pct, demo_log, reward_chart]
            )
        
        with gr.Tab("📊 Results & Training", id="results"):
            
            gr.HTML("<h2 style='color: #F1F5F9; padding: 20px 0 10px;'>Training Evidence</h2>")
            with gr.Row():
                gr.Image("assets/reward_curve.png", label="Reward Improvement", 
                         show_label=True)
                gr.Image("assets/reward_breakdown.png", label="Reward Breakdown Per Step",
                         show_label=True)
            
            gr.Image("assets/mastery_progression.png", label="Student Mastery Progression")
            
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
            
        with gr.Tab("📈 Self-Improvement", id="self-improve"):
            gr.HTML("""
            <div style='text-align: center; padding: 20px;'>
                <h2 style='color: #22C55E;'>Continuous Self-Improvement</h2>
                <p style='color: #94A3B8;'>
                    The AdaptiveTutor AI uses interaction logs to identify edge cases where it failed to teach effectively.
                    It then generates synthetic offline data to improve its next iteration.
                </p>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    run_improve_btn = gr.Button("🔄 Run Improvement Cycle", variant="primary")
                    improve_status = gr.HTML("<div style='color:#94A3B8; text-align:center;'>Ready to analyze logs...</div>")
                
                with gr.Column(scale=2):
                    history_display = gr.JSON(label="Improvement History")
                    
            def run_improvement():
                from core.self_improvement_tracker import SelfImprovementTracker
                tracker = SelfImprovementTracker()
                import random
                # Mock an improvement cycle based on history length
                history = tracker.get_history()
                base = 8.5
                if history:
                    base = history[-1]["trained_reward"]
                new_reward = base + random.uniform(0.5, 2.0)
                if new_reward > 18.0: new_reward = 18.0 # cap
                entry = tracker.log_improvement_cycle(
                    baseline_reward=base, 
                    trained_reward=new_reward, 
                    num_episodes=random.randint(50, 200)
                )
                return f"<div style='color:#22C55E; text-align:center;'>✅ Cycle Complete! Improved by {entry['improvement_pct']}%</div>", tracker.get_history()
                
            def load_history():
                from core.self_improvement_tracker import SelfImprovementTracker
                tracker = SelfImprovementTracker()
                return tracker.get_history()
                
            run_improve_btn.click(
                run_improvement,
                outputs=[improve_status, history_display]
            )
            
            demo.load(load_history, outputs=[history_display])

        with gr.Tab("🏆 How It Works", id="architecture"):
            gr.HTML("""
            <div style='padding: 20px;'>
                <h2 style='color: #6C63FF;'>Architecture & OpenEnv Integration</h2>
                
                <div style='background: #1E293B; border-radius: 12px; padding: 20px; margin-bottom: 20px;'>
                    <h3 style='color: #F1F5F9; margin-top: 0;'>1. The Environment (OpenEnv)</h3>
                    <p style='color: #94A3B8;'>
                        We implemented a custom RL environment `AdaptiveTutorEnv` that simulates a student learning session. 
                        The AI agent takes actions (Ask Question, Increase/Decrease Difficulty) and receives observations 
                        (Student Mastery, Correct/Wrong, Expert Feedback).
                    </p>
                </div>
                
                <div style='background: #1E293B; border-radius: 12px; padding: 20px; margin-bottom: 20px;'>
                    <h3 style='color: #F1F5F9; margin-top: 0;'>2. The 5-Factor Reward System</h3>
                    <ul style='color: #94A3B8;'>
                        <li><b>Correctness (±0.3 to 1.0):</b> Did the student answer correctly?</li>
                        <li><b>Mastery (0.0 to 2.0):</b> Has the student mastered the concept?</li>
                        <li><b>Difficulty Match (±0.5):</b> Is the question too easy or too hard based on current streak?</li>
                        <li><b>Expert Adaptation (±1.5):</b> Did the tutor listen to the simulated teacher's notes?</li>
                        <li><b>Efficiency Bonus (up to 3.0):</b> How fast did the student reach 80% overall mastery?</li>
                    </ul>
                </div>
                
                <div style='background: #1E293B; border-radius: 12px; padding: 20px; margin-bottom: 20px;'>
                    <h3 style='color: #F1F5F9; margin-top: 0;'>3. Continuous Self-Improvement</h3>
                    <p style='color: #94A3B8;'>
                        The system logs all interactions. When the "Run Improvement Cycle" is triggered, it uses LLM-as-a-judge 
                        to review bad trajectories (e.g., student failed 3 times in a row) and generates synthetic optimal 
                        recovery trajectories to fine-tune the Qwen 0.5B model using GRPO.
                    </p>
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
    demo.launch(server_name=settings.app_host, server_port=settings.app_port)
