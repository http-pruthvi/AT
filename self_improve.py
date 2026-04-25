"""
self_improve.py

Builds training-ready data from tutoring interaction logs and prints
progress metrics for iterative model improvement.
"""

import glob
import json
import os
from collections import defaultdict


LOGS_DIR = "logs"
OUTPUT_DIR = "outputs"
OUTPUT_DATASET = os.path.join(OUTPUT_DIR, "self_improve_dataset.jsonl")
OUTPUT_METRICS = os.path.join(OUTPUT_DIR, "self_improve_metrics.json")


def _iter_log_records():
    for path in glob.glob(os.path.join(LOGS_DIR, "*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    per_student = defaultdict(lambda: {"turns": 0, "correct": 0, "gain_sum": 0.0})

    for r in _iter_log_records():
        subject = r.get("subject", "General")
        question = r.get("question", "")
        student_answer = r.get("student_answer", "")
        correct_answer = r.get("correct_answer", "")
        is_correct = bool(r.get("is_correct", False))
        concept = r.get("concept", "general")
        difficulty = r.get("difficulty", "medium")

        prompt = (
            f"<|system|>\n"
            f"You are an adaptive tutor improving from past interactions.\n"
            f"Generate a better next question for the student.\n"
            f"</s>\n"
            f"<|user|>\n"
            f"Subject: {subject}\n"
            f"Concept: {concept}\n"
            f"Difficulty: {difficulty}\n"
            f"Previous question: {question}\n"
            f"Student answer: {student_answer}\n"
            f"Was correct: {is_correct}\n"
            f"Target answer knowledge: {correct_answer}\n"
            f"Generate ONE improved follow-up question only.\n"
            f"</s>\n"
            f"<|assistant|>"
        )

        rows.append({"prompt": prompt})

        student = r.get("student_name", "unknown")
        gain = float(r.get("learning_metrics", {}).get("learning_gain_pct", 0.0))
        per_student[student]["turns"] += 1
        per_student[student]["correct"] += int(is_correct)
        per_student[student]["gain_sum"] += gain

    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    metrics = {
        "total_turns": len(rows),
        "students": len(per_student),
        "per_student": {},
    }
    for student, v in per_student.items():
        turns = max(1, v["turns"])
        metrics["per_student"][student] = {
            "turns": v["turns"],
            "accuracy_pct": round((v["correct"] / turns) * 100, 1),
            "avg_learning_gain_pct": round(v["gain_sum"] / turns, 2),
        }

    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    result = build_dataset()
    print("Self-improvement artifacts generated:")
    print(f"- Dataset: {OUTPUT_DATASET}")
    print(f"- Metrics: {OUTPUT_METRICS}")
    print(f"- Turns: {result['total_turns']} | Students: {result['students']}")
