# store_results.py - Auto-generated
import json

def save(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f" Results saved to {path}")