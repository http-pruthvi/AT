import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AdaptiveTutor AI — GRPO Training\n",
    "\n",
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/http-pruthvi/adaptive-tutor-ai/blob/main/training.ipynb)\n",
    "\n",
    "**Meta PyTorch OpenEnv Hackathon Grand Finale — April 2026**\n",
    "\n",
    "This notebook trains an AI tutor using GRPO (Group Relative Policy Optimization) via TRL.\n",
    "The environment is running live on HuggingFace Spaces, so there's nothing to install locally.\n",
    "\n",
    "We hit the API, collect trajectories, and use the rewards to train a small LLM to make\n",
    "better tutoring decisions."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Install everything we need\n",
    "torch is sometimes missing on fresh Colab runtimes, so we install it explicitly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118\n",
    "%pip install -q trl transformers datasets accelerate bitsandbytes\n",
    "%pip install -q requests matplotlib pandas pydantic"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Connect to the live environment\n",
    "Our OpenEnv server is deployed on HuggingFace Spaces. We just talk to it over HTTP."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import requests, json, random, time\n",
    "\n",
    "BASE_URL = 'https://http-pruthvi-adaptive-tutor-ai.hf.space/api'\n",
    "\n",
    "print(\"Waking up HF Space...\")\n",
    "for i in range(3):\n",
    "    try:\n",
    "        r = requests.get(f\"{BASE_URL}/health\", timeout=30)\n",
    "        if r.status_code == 200:\n",
    "            print(\"Space is awake!\")\n",
    "            break\n",
    "    except:\n",
    "        print(f\"Attempt {i+1} failed, waiting 10s...\")\n",
    "        time.sleep(10)\n",
    "\n",
    "def env_reset():\n",
    "    \"\"\"Start a fresh tutoring episode.\"\"\"\n",
    "    r = requests.post(f'{BASE_URL}/reset', timeout=30)\n",
    "    r.raise_for_status()\n",
    "    data = r.json()\n",
    "    return data.get('observation', data)\n",
    "\n",
    "def env_step(action):\n",
    "    \"\"\"Send an action, get back observation + reward.\"\"\"\n",
    "    r = requests.post(f'{BASE_URL}/step', json={'action': action}, timeout=30)\n",
    "    r.raise_for_status()\n",
    "    data = r.json()\n",
    "    obs = data.get('observation', data)\n",
    "    reward = data.get('reward', obs.get('reward', 0))\n",
    "    done = data.get('done', obs.get('done', False))\n",
    "    info = data.get('info', obs.get('info', {}))\n",
    "    return obs, reward, done, info\n",
    "\n",
    "# --- Verification Tests ---\n",
    "print(\"\\nRunning Verification Tests...\")\n",
    "health = requests.get(f\"{BASE_URL}/health\").json()\n",
    "print(f\"Health: {health}\")\n",
    "\n",
    "reset_obs = env_reset()\n",
    "print(f\"Reset Success! Subject: {reset_obs.get('current_topic')}\")\n",
    "\n",
    "step_obs, reward, done, _ = env_step({\n",
    "    \"action_type\": \"ask_question\",\n",
    "    \"target_concept\": \"algebra\",\n",
    "    \"difficulty\": 1\n",
    "})\n",
    "print(f\"Step Success! Reward: {reward}\")\n",
    "\n",
    "print('\\nEnvironment is ready for training!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Baseline: heuristic agent\n",
    "Before we train anything, let's see how a simple rule-based agent does.\n",
    "This gives us a number to beat."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def heuristic_action(obs):\n",
    "    \"\"\"Simple rules: lower difficulty when struggling, raise when on a streak.\"\"\"\n",
    "    p = obs.get('student_profile', {})\n",
    "    w = p.get('weakest_concept', '')\n",
    "    cw = obs.get('consecutive_wrong', 0)\n",
    "    cc = obs.get('consecutive_correct', 0)\n",
    "    d = obs.get('current_difficulty', 1)\n",
    "    if cw >= 2 and d > 1:\n",
    "        return {'action_type': 'decrease_difficulty', 'content': '', 'target_concept': w, 'difficulty': max(1, d-1)}\n",
    "    if cc >= 3:\n",
    "        return {'action_type': 'increase_difficulty', 'content': '', 'target_concept': w, 'difficulty': min(5, d+1)}\n",
    "    return {'action_type': 'ask_question', 'content': '', 'target_concept': w, 'difficulty': d}\n",
    "\n",
    "def run_episode(agent_fn):\n",
    "    \"\"\"Run a full 20-step episode and return total reward.\"\"\"\n",
    "    obs = env_reset()\n",
    "    total, rewards = 0.0, []\n",
    "    for _ in range(20):\n",
    "        action = agent_fn(obs)\n",
    "        obs, reward, done, info = env_step(action)\n",
    "        total += reward\n",
    "        rewards.append(reward)\n",
    "        if done: break\n",
    "    return total, rewards\n",
    "\n",
    "# run 5 episodes to get a stable baseline\n",
    "baseline_totals = []\n",
    "for i in range(5):\n",
    "    total, _ = run_episode(heuristic_action)\n",
    "    baseline_totals.append(total)\n",
    "    print(f'Baseline episode {i+1}: {total:.1f}')\n",
    "\n",
    "avg_baseline = sum(baseline_totals) / len(baseline_totals)\n",
    "print(f'\\nBaseline average: {avg_baseline:.1f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Load the model\n",
    "Qwen2.5-0.5B-Instruct — ultra-fast and efficient.\n",
    "We'll fine-tune it to make better tutoring decisions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
    "\n",
    "MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(MODEL)\n",
    "tokenizer.pad_token = tokenizer.eos_token\n",
    "\n",
    "model = AutoModelForCausalLM.from_pretrained(\n",
    "    MODEL, torch_dtype=torch.float16, device_map='auto'\n",
    ")\n",
    "print(f'Loaded {MODEL}')\n",
    "print(f'{model.num_parameters()/1e6:.0f}M parameters on {model.device}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Generate training prompts\n",
    "We create 60 diverse prompts by resetting the environment and stepping\n",
    "to random states. This way the model sees a variety of student situations."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import Dataset\n",
    "\n",
    "def build_prompt(obs):\n",
    "    p = obs.get('student_profile', {})\n",
    "    weak = ', '.join(p.get('weak_concepts', [])[:3]) or 'none yet'\n",
    "    expert = obs.get('expert_feedback') or 'No feedback'\n",
    "    return (f'You are an AI tutor deciding what to do next.\\n'\n",
    "            f'Subject: {obs.get(\"current_topic\", \"?\")}\\n'\n",
    "            f'Difficulty: {obs.get(\"current_difficulty\", 1)}/5\\n'\n",
    "            f'Last answer correct: {obs.get(\"student_correct\", False)}\\n'\n",
    "            f'Correct streak: {obs.get(\"consecutive_correct\", 0)}\\n'\n",
    "            f'Wrong streak: {obs.get(\"consecutive_wrong\", 0)}\\n'\n",
    "            f'Overall mastery: {p.get(\"overall_mastery\", 0):.0%}\\n'\n",
    "            f'Weak areas: {weak}\\n'\n",
    "            f'Expert feedback: {expert}\\n\\n'\n",
    "            f'Respond with a JSON action: {{\"action_type\": \"ask_question|give_hint|explain_concept|increase_difficulty|decrease_difficulty\", \"target_concept\": \"...\", \"difficulty\": 1-5}}')\n",
    "\n",
    "prompts = []\n",
    "for i in range(60):\n",
    "    obs = env_reset()\n",
    "    # step a random number of times to get varied states\n",
    "    for _ in range(random.randint(0, 8)):\n",
    "        obs, _, done, _ = env_step(heuristic_action(obs))\n",
    "        if done: break\n",
    "    prompts.append({'prompt': build_prompt(obs)})\n",
    "    if (i+1) % 20 == 0:\n",
    "        print(f'Generated {i+1}/60 prompts')\n",
    "\n",
    "dataset = Dataset.from_list(prompts)\n",
    "print(f'\\nDataset ready: {len(dataset)} prompts')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Define the reward function\n",
    "This is where the magic happens. For each LLM completion, we parse it as\n",
    "a tutoring action, run 5 steps against the live environment, and return\n",
    "the cumulative reward. That becomes the GRPO training signal."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def parse_action(text, obs):\n",
    "    \"\"\"Try to pull a JSON action out of the LLM's response.\"\"\"\n",
    "    try:\n",
    "        s, e = text.find('{'), text.rfind('}') + 1\n",
    "        if s >= 0 and e > s:\n",
    "            a = json.loads(text[s:e])\n",
    "            return {\n",
    "                'action_type': a.get('action_type', 'ask_question'),\n",
    "                'content': a.get('content', ''),\n",
    "                'target_concept': a.get('target_concept', ''),\n",
    "                'difficulty': max(1, min(5, int(a.get('difficulty', 1))))\n",
    "            }\n",
    "    except:\n",
    "        pass\n",
    "    # if parsing fails, fall back to the heuristic\n",
    "    return heuristic_action(obs)\n",
    "\n",
    "def reward_fn(completions, **kwargs):\n",
    "    \"\"\"GRPO reward: run each completion against the environment.\"\"\"\n",
    "    rewards = []\n",
    "    for comp in completions:\n",
    "        text = comp if isinstance(comp, str) else comp[0].get('content', '')\n",
    "        obs = env_reset()\n",
    "        total = 0.0\n",
    "        for _ in range(5):\n",
    "            action = parse_action(text, obs)\n",
    "            obs, r, done, _ = env_step(action)\n",
    "            total += r\n",
    "            if done: break\n",
    "        rewards.append(total)\n",
    "    return rewards\n",
    "\n",
    "print('Reward function ready')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Train with GRPO\n",
    "Conservative settings so it actually finishes on Colab free tier.\n",
    "1 epoch, batch size 2, 4 generations per prompt."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from trl import GRPOConfig, GRPOTrainer\n",
    "\n",
    "config = GRPOConfig(\n",
    "    output_dir='./grpo_output',\n",
    "    num_train_epochs=1,\n",
    "    per_device_train_batch_size=2,\n",
    "    gradient_accumulation_steps=4,\n",
    "    learning_rate=5e-6,\n",
    "    max_completion_length=128,\n",
    "    num_generations=4,\n",
    "    logging_steps=5,\n",
    "    save_steps=50,\n",
    "    report_to='none',\n",
    ")\n",
    "\n",
    "trainer = GRPOTrainer(\n",
    "    model=model,\n",
    "    config=config,\n",
    "    tokenizer=tokenizer,\n",
    "    train_dataset=dataset,\n",
    "    reward_funcs=reward_fn,\n",
    ")\n",
    "\n",
    "print('Starting GRPO training...')\n",
    "trainer.train()\n",
    "print('Done!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Evaluate the trained model\n",
    "Same setup as baseline — 5 episodes, but now using the trained model's decisions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def trained_action(obs):\n",
    "    prompt = build_prompt(obs)\n",
    "    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)\n",
    "    with torch.no_grad():\n",
    "        out = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)\n",
    "    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)\n",
    "    return parse_action(text, obs)\n",
    "\n",
    "trained_totals = []\n",
    "for i in range(5):\n",
    "    total, _ = run_episode(trained_action)\n",
    "    trained_totals.append(total)\n",
    "    print(f'Trained episode {i+1}: {total:.1f}')\n",
    "\n",
    "avg_trained = sum(trained_totals) / len(trained_totals)\n",
    "print(f'\\nTrained average: {avg_trained:.1f}')\n",
    "print(f'Baseline average: {avg_baseline:.1f}')\n",
    "print(f'Improvement: {avg_trained - avg_baseline:+.1f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Plot the results\n",
    "Side-by-side comparison for the judges."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))\n",
    "\n",
    "# averages\n",
    "bars = ax1.bar(['Baseline\\n(Heuristic)', 'Trained\\n(GRPO)'],\n",
    "               [avg_baseline, avg_trained],\n",
    "               color=['#667eea', '#4ade80'], width=0.5, edgecolor='white', lw=1.5)\n",
    "for b, v in zip(bars, [avg_baseline, avg_trained]):\n",
    "    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{v:.1f}',\n",
    "             ha='center', fontweight='bold', fontsize=14)\n",
    "ax1.set_ylabel('Avg Episode Reward')\n",
    "ax1.set_title('Before vs After Training', fontweight='bold')\n",
    "ax1.set_ylim(0, max(avg_baseline, avg_trained)*1.3)\n",
    "ax1.grid(axis='y', alpha=0.3)\n",
    "\n",
    "# per-episode\n",
    "ax2.plot(range(1,6), baseline_totals, 'o-', color='#667eea', label='Baseline', lw=2, ms=8)\n",
    "ax2.plot(range(1,6), trained_totals, 's-', color='#4ade80', label='Trained', lw=2, ms=8)\n",
    "ax2.axhline(avg_baseline, color='#667eea', ls='--', alpha=0.4)\n",
    "ax2.axhline(avg_trained, color='#4ade80', ls='--', alpha=0.4)\n",
    "ax2.set_xlabel('Episode')\n",
    "ax2.set_ylabel('Total Reward')\n",
    "ax2.set_title('Per-Episode Comparison', fontweight='bold')\n",
    "ax2.legend()\n",
    "ax2.grid(alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('reward_comparison.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print('Saved to reward_comparison.png')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Push to HuggingFace Hub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "REPO = 'http-pruthvi/adaptive-tutor-qwen-grpo'\n",
    "\n",
    "model.push_to_hub(REPO)\n",
    "tokenizer.push_to_hub(REPO)\n",
    "print(f'Pushed to https://huggingface.co/{REPO}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('training/training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
