import os
import json
import time

class SelfImprovementTracker:
    def __init__(self, log_path="data/improvement_log.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._ensure_log_exists()
        
    def _ensure_log_exists(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)
                
    def log_improvement_cycle(self, baseline_reward: float, trained_reward: float, num_episodes: int):
        with open(self.log_path, 'r') as f:
            logs = json.load(f)
            
        improvement_pct = 0.0
        if baseline_reward != 0:
            improvement_pct = ((trained_reward - baseline_reward) / abs(baseline_reward)) * 100
            
        entry = {
            "cycle_id": len(logs) + 1,
            "timestamp": time.time(),
            "baseline_reward": round(baseline_reward, 2),
            "trained_reward": round(trained_reward, 2),
            "improvement_pct": round(improvement_pct, 2),
            "num_episodes": num_episodes
        }
        
        logs.append(entry)
        
        with open(self.log_path, 'w') as f:
            json.dump(logs, f, indent=2)
            
        return entry
        
    def get_history(self):
        with open(self.log_path, 'r') as f:
            return json.load(f)
