# Necessary libraries for the pipeline. Ensure these are installed before running the scripts.
# !pip install catboost folktables scikit-learn pandas numpy mlxtend


import subprocess
import sys
import time
from pathlib import Path


def run_script(script_name):
    """ Helper function to execute scripts via subprocess. """
    print(f"\n{'='*50}")
    print(f"Executing: {script_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    try:
        subprocess.run([sys.executable, script_name], check=True, text=True)
        elapsed_time = time.time() - start_time
        print(f"\nTask completed in {elapsed_time:.2f}s")
        
    except subprocess.CalledProcessError:
        print(f"\nError during execution of {script_name}. Pipeline aborted.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\nScript {script_name} not found in the current directory.")
        sys.exit(1)


if __name__ == "__main__":
    print("Starting the ML Explainability & Fairness Pipeline...")
    total_start_time = time.time()
    
    current_dir = Path.cwd()
    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset Generation and Preprocessing
    run_script("create_dataset.py")
    
    # 2. Model Training and Counterfactual Extraction
    run_script("feature_importance.py")
    
    # 3. Association Rules Mining and Bias Auditing
    run_script("data_mining.py")
    
    total_time = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(f"Pipeline finished successfully in {total_time:.2f}s")
    print(f"Results, reports, and plots are available in the '{data_dir}' folder.")