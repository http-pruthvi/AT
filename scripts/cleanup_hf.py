import os
from huggingface_hub import HfApi

def cleanup():
    api = HfApi()
    repo_id = "http-pruthvi/adaptive-tutor-ai"
    
    # Files that should NO LONGER be in the root
    files_to_delete = [
        "student_model.py",
        "expert_simulator.py",
        "reward.py",
        "question_generator.py",
        "session_manager.py",
        "product_evaluator.py",
        "train.py",
        "training.ipynb",
        "self_improve.py",
        "pitch.md",
        "train_log.txt",
        "mastery_progression.png",
        "reward_breakdown.png",
        "reward_curve.png"
    ]
    
    print(f"Cleaning up {repo_id}...")
    
    for file in files_to_delete:
        try:
            api.delete_file(path_in_repo=file, repo_id=repo_id, repo_type="space")
            print(f"Deleted: {file}")
        except Exception as e:
            # If file doesn't exist, ignore
            if "404" in str(e):
                print(f"Skipped (not found): {file}")
            else:
                print(f"Error deleting {file}: {e}")

    # Also delete the old subjects folder if it was duplicated (it shouldn't be if it's still in root locally)
    # But wait, subjects IS still in root locally.
    
    print("Cleanup complete!")

if __name__ == "__main__":
    cleanup()
