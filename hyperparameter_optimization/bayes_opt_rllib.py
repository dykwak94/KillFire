# Bayesian Optimization
# Find the best parameter that makes the return curve increase steadily
# Last Update : 2025.06.01 (TRAIN_SCRIPT_PATH changed)
import os
import pandas as pd
import optuna
import subprocess
import re

# === PATHS ===
FOREST_ENV_PATH = os.path.join(os.path.dirname(__file__), "../environment/forest_env.py")
TRAIN_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "../train/train_rllib.py")
RESULTS_DIR = "../results/"

def modify_forest_env(suppressed, onfire):
    with open(FOREST_ENV_PATH, "r") as f:
        code = f.read()
    code = re.sub(r'SUPPRESSED_COEFF\s*=\s*\d+(\.\d+)?', f'SUPPRESSED_COEFF = {suppressed}', code)
    code = re.sub(r'ONFIRE_COEFF\s*=\s*\d+(\.\d+)?', f'ONFIRE_COEFF = {onfire}', code)
    with open(FOREST_ENV_PATH, "w") as f:
        f.write(code)

def get_latest_progress_csv(results_dir=RESULTS_DIR):
    # Find all subdirectories in results_dir
    run_folders = [os.path.join(results_dir, d) for d in os.listdir(results_dir)
                   if os.path.isdir(os.path.join(results_dir, d))]
    if not run_folders:
        raise FileNotFoundError(f"No run folders found in {results_dir}")

    # Latest modified run folder
    latest_run_folder = max(run_folders, key=os.path.getmtime)

    # Expect exactly one subfolder inside the run folder
    subfolders = [os.path.join(latest_run_folder, d) for d in os.listdir(latest_run_folder)
                  if os.path.isdir(os.path.join(latest_run_folder, d))]
    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in {latest_run_folder}")
    if len(subfolders) > 1:
        print(f"Warning: Multiple subfolders in {latest_run_folder}, using the first one.")

    final_folder = subfolders[0]
    progress_csv_path = os.path.join(final_folder, "progress.csv")
    if not os.path.exists(progress_csv_path):
        raise FileNotFoundError(f"progress.csv not found at {progress_csv_path}")
    return progress_csv_path

def calculate_combined_reward(df, final_n=10, alpha=1.0, beta=1.0):
    # Support both column names for RLlib reward
    '''
    # 20250530 0951
    for col in ["env_runners/episode_reward_mean", "episode_reward_mean"]:
        if col in df.columns:
            auc = df[col].sum()
            final_reward = df[col].tail(final_n).mean()
            return alpha * auc + beta * final_reward
    print(f"Reward column not found! Columns available: {df.columns}")
    return float('-inf')  # or np.nan or raise Exception if you prefer
    '''
    # 20250530 1736
    for col in ["env_runners/episode_reward_mean", "episode_reward_mean"]:
        if col in df.columns:
            inital_reward = df[col].head(final_n).mean()
            final_reward = df[col].tail(final_n).mean()
            return final_reward - inital_reward
    print(f"Reward column not found! Columns available: {df.columns}")
    return float('-inf')  # or np.nan or raise Exception if you prefer


def run_training():
    env = os.environ.copy()
    env['PYTHONPATH'] = '..'
    env['ITERATION'] = str(1000)
    subprocess.run(
        ["python", TRAIN_SCRIPT_PATH],
        env=env,
        check=True,
    )

def objective(trial):
    # === SAMPLE HYPERPARAMETERS ===
    suppressed = trial.suggest_float("SUPPRESSED_COEFF", 1, 1000)
    onfire = trial.suggest_float("ONFIRE_COEFF", 1, 10)

    modify_forest_env(suppressed, onfire)
    run_training()
    progress_csv = get_latest_progress_csv()
    df = pd.read_csv(progress_csv)
    reward = calculate_combined_reward(df, final_n=10, alpha=0.2, beta=1.0)
    print("SUPPRESSED:", suppressed)
    print("ONFIRE:", onfire)
    return reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Adjust as needed

    print("Best parameters:", study.best_params)
    print("Best combined reward (AUC + final):", study.best_value)

    # Print and save all trial results
    df = study.trials_dataframe()
    #print(df.head)
    print(df[["number", "value", "params_SUPPRESSED_COEFF", "params_ONFIRE_COEFF"]])
    df.to_csv("optuna_trials.csv", index=False)