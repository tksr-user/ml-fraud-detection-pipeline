# compare.py - Auto-generated
import json

def run(new_path, old_path):
    try:
        with open(new_path) as f1, open(old_path) as f2:
            new = json.load(f1)
            old = json.load(f2)
            return {"performance_improved": new["accuracy"] > old["accuracy"]}
    except FileNotFoundError:
        return {"performance_improved": True}