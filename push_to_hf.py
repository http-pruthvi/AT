import os
from huggingface_hub import HfApi

def push_to_hf():
    # Configuration
    repo_id = "http-pruthvi/adaptive-tutor-ai"
    local_dir = "." # Push everything in the root
    
    # Files/folders to exclude
    exclude = [
        ".git",
        "__pycache__",
        "adaptive_tutor_trained", # Don't push local checkpoints yet
        "venv",
        "outputs",
        "*.log",
        "train_log.txt"
    ]
    
    print(f"🚀 Syncing local files to Hugging Face Space: {repo_id}")
    
    try:
        api = HfApi()
        # Note: This requires HUGGING_FACE_HUB_TOKEN to be set in environment
        # or for the user to be logged in via `huggingface-cli login`
        api.upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=exclude,
            delete_patterns=None # Don't delete things in the Space that aren't local
        )
        print("✅ Successfully pushed to Hugging Face!")
    except Exception as e:
        print(f"❌ Error pushing to HF: {e}")
        print("\nTip: Make sure you are logged in using 'huggingface-cli login' or have set the HUGGING_FACE_HUB_TOKEN environment variable.")

if __name__ == "__main__":
    push_to_hf()
