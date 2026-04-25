"""
product_evaluator.py - Evaluates human student answers using LLMs (or fallback).
"""
import json
import httpx

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"

class ProductEvaluator:
    def __init__(self):
        from shared import AI_LOADED, ai_model, ai_tokenizer
        self.ai_model = ai_model
        self.ai_tokenizer = ai_tokenizer
        self.ai_loaded = AI_LOADED
        self.ollama_available = self.check_ollama()

    def check_ollama(self):
        """Ping Ollama to see if it's up and has our model."""
        try:
            r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return any(OLLAMA_MODEL.split(":")[0] in m for m in models)
        except Exception:
            pass
        return False

    def evaluate_answer(self, subject: str, question: str, correct_answer: str, student_answer: str) -> dict:
        """Evaluate the student's answer using Ollama, local model, or fallback."""
        if self.ollama_available:
            return self._evaluate_with_ollama(subject, question, correct_answer, student_answer)
        elif self.ai_loaded:
            return self._evaluate_with_local_model(subject, question, correct_answer, student_answer)
        else:
            return self._evaluate_with_fallback(correct_answer, student_answer)

    def _evaluate_with_local_model(self, subject: str, question: str, correct_answer: str, student_answer: str) -> dict:
        """Evaluate using the shared local Qwen model."""
        prompt = f"""<|system|>
You are a {subject} teacher. Evaluate the student's answer simply.
Respond in JSON only: {{"is_correct": true/false, "explanation": "simple text", "encouragement": "motivating message", "mastery_delta": 0.1}}
</s>
<|user|>
Question: {question}
Correct Answer: {correct_answer}
Student Answer: {student_answer}
</s>
<|assistant|>
"""
        try:
            import torch
            inputs = self.ai_tokenizer(prompt, return_tensors="pt").to(self.ai_model.device)
            with torch.no_grad():
                outputs = self.ai_model.generate(**inputs, max_new_tokens=128, temperature=0.1)
            resp_text = self.ai_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Simple JSON extraction
            start = resp_text.find("{")
            end = resp_text.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(resp_text[start:end])
                return {
                    "is_correct": bool(data.get("is_correct", False)),
                    "explanation": str(data.get("explanation", "Good try!")),
                    "encouragement": str(data.get("encouragement", "Keep it up!")),
                    "mastery_delta": float(data.get("mastery_delta", 0.05))
                }
        except:
            pass
        return self._evaluate_with_fallback(correct_answer, student_answer)

    def _evaluate_with_ollama(self, subject: str, question: str, correct_answer: str, student_answer: str) -> dict:
        prompt = f"""You are a {subject} teacher who specializes in explaining complex things simply.
Question asked: {question}
Correct answer: {correct_answer}
Student answered: {student_answer}

Evaluate if the student is correct.
Be lenient with spelling and phrasing.

Respond in JSON only:
{{
  "is_correct": true/false,
  "explanation": "Brief, VERY SIMPLE explanation of why, avoiding jargon",
  "encouragement": "A short, friendly and motivating message",
  "mastery_delta": 0.05 to 0.15
}}"""

        try:
            # Use format="json" if the model supports it, but Qwen2.5 is smart enough anyway
            r = httpx.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1} # low temp for evaluation
                },
                timeout=15.0,
            )
            if r.status_code == 200:
                resp_text = r.json().get("response", "")
                
                # extract json if it's wrapped in markdown blocks
                start = resp_text.find("{")
                end = resp_text.rfind("}") + 1
                if start != -1 and end > start:
                    data = json.loads(resp_text[start:end])
                    
                    # Cap the mastery delta just to be safe
                    delta = float(data.get("mastery_delta", 0.05))
                    delta = max(-0.15, min(0.15, delta))
                    
                    return {
                        "is_correct": bool(data.get("is_correct", False)),
                        "explanation": str(data.get("explanation", "Could not parse explanation.")),
                        "encouragement": str(data.get("encouragement", "Keep going!")),
                        "mastery_delta": delta
                    }
        except Exception as e:
            print(f"Ollama eval failed: {e}")
            pass
        
        # If anything fails, use the fallback
        return self._evaluate_with_fallback(correct_answer, student_answer)

    def _evaluate_with_fallback(self, correct_answer: str, student_answer: str) -> dict:
        """Simple keyword matching fallback for when Ollama is offline."""
        target_words = set(correct_answer.lower().replace('.', '').replace(',', '').split())
        student_words = set(student_answer.lower().replace('.', '').replace(',', '').split())
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'of', 'in', 'to', 'and', 'it', 'that', 'this'}
        target_words = target_words - stop_words
        student_words = student_words - stop_words
        
        if not target_words:
            is_correct = True # edge case
        else:
            match_ratio = len(target_words.intersection(student_words)) / len(target_words)
            is_correct = match_ratio >= 0.5
            
        if is_correct:
            return {
                "is_correct": True,
                "explanation": f"Good job! You got the key concepts.",
                "encouragement": "You're doing great, keep it up!",
                "mastery_delta": 0.10
            }
        else:
            target_str = " ".join(target_words) if target_words else correct_answer
            return {
                "is_correct": False,
                "explanation": f"Not quite. The answer we were looking for involves: {target_str}.",
                "encouragement": "Don't worry, let's learn from this and try another one.",
                "mastery_delta": -0.05
            }

if __name__ == "__main__":
    # Quick test
    evaluator = ProductEvaluator()
    res1 = evaluator.evaluate_answer(
        "History", 
        "Who was the first president of the United States?", 
        "George Washington", 
        "washington"
    )
    print("Test 1 (Correct):", res1)
    
    res2 = evaluator.evaluate_answer(
        "Math", 
        "What is 5 x 5?", 
        "25", 
        "20"
    )
    print("Test 2 (Wrong):", res2)
