00:00-00:10 HOOK:
"LLMs are trained to give the right answer.
I built an OpenEnv environment that trains them 
to ask the right question."

00:10-00:45 PROBLEM + SOLUTION:
"Hi, I'm Pruthvi. AI tutors today act like encyclopedias.
Student struggles? AI gives the answer. That's not teaching.
AdaptiveTutor AI is an RL environment that forces an LLM 
to learn scaffolding — through a 5-dimensional reward signal:
correctness, mastery growth, difficulty calibration, 
efficiency, and expert alignment."

00:45-01:45 DEMO + SNORKEL BONUS:
"Let me show you Human Mode — a real student at a keyboard.
[type wrong answer - show difficulty dropping]
[type right answer - show mastery bar filling]
Now Demo Mode — watch the RL simulation.
[switch to demo, show reward counter going up]
The Snorkel AI bonus: every 5 steps, our Expert Simulator 
shifts preferences. Dr. Sharma switches from proof-based 
to conceptual math. The trained LLM detects this drift 
and adapts its policy. It's optimizing against a moving target."

01:45-02:30 RESULTS + SOLO ADVANTAGE:
"Results: Random policy averages 8.5 reward.
After GRPO training on this environment: 14.60. +72%.
[show reward_curve.png]
As a solo builder, I had zero architectural friction.
Every OpenEnv reward state maps directly to a UI element.
The mastery bar IS the mastery_reward function. 
One unified system."

02:30-03:00 CLOSE:
"300 million students in India.
We've given them access to AI.
We haven't given them a tutor.
AdaptiveTutor AI proves we can train LLMs to actually teach.
The environment is live, the training pipeline works,
and the product is ready.
Thank you."

Q&A PREP (memorize these):

Q: "How is this different from just prompting GPT-4 to tutor?"
A: "GPT-4 is a static model. It cannot improve its teaching 
    strategy based on whether the student actually learned.
    This environment generates training signal — reward — 
    that updates the model weights. The model gets measurably 
    better at teaching over episodes. That's RL, not prompting."

Q: "How do you know the agent actually learned and didn't just 
    memorize the question bank?"
A: "The student model uses probabilistic mastery — not fixed answers.
    The same question gets different outcomes based on hidden mastery state.
    The agent must learn the STRATEGY of teaching, not the answers."

Q: "Why not just use human feedback instead of the expert simulator?"
A: "Human feedback is expensive and slow for RL training.
    The expert simulator gives us 1000x more training signal.
    Once the model is trained here, we deploy to real students
    and fine-tune with actual human feedback. 
    Simulation first, real world second."
