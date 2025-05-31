# hyperparam_search.py
# For grid search
# Last Update : 2025.05.29
import subprocess
import itertools
import re
import os

FOREST_ENV_PATH = "../environment/forest_env.py"
TRAIN_SCRIPT_PATH = "../train/train_rllib.py"
SUPPRESSED_COEFFS = [1] + list(range(100, 1001, 100))
ONFIRE_COEFFS = list(range(1, 11))
ITERATION = 1000

def modify_forest_env(suppressed, onfire):
    with open(FOREST_ENV_PATH, "r") as f:
        code = f.read()
    code = re.sub(r'SUPPRESSED_COEFF\s*=\s*\d+', f'SUPPRESSED_COEFF = {suppressed}', code)
    code = re.sub(r'ONFIRE_COEFF\s*=\s*\d+', f'ONFIRE_COEFF = {onfire}', code)
    with open(FOREST_ENV_PATH, "w") as f:
        f.write(code)

def run_training():
    env = dict(os.environ)
    env['PYTHONPATH'] = '..'
    env['ITERATION'] = str(ITERATION)
    subprocess.run(
        ["python", TRAIN_SCRIPT_PATH],
        env=env
    )

def main():
    for suppressed, onfire in itertools.product(SUPPRESSED_COEFFS, ONFIRE_COEFFS):
        print(f"SUPPRESSED_COEFF={suppressed}, ONFIRE_COEFF={onfire}")
        modify_forest_env(suppressed, onfire)
        run_training()

if __name__ == "__main__":
    main()