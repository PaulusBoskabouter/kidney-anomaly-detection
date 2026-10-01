import os
import subprocess
import sys


def main():
    TOKENS = {}
    with open("TOKENS.csv", 'r') as file:
        for line in file:
            line = line.split(',')
            TOKENS[line[0]] = line[1].rstrip('\n')

    for model_name, token in TOKENS.items():
        env = os.environ.copy()
        env["HF_TOKEN"] = token

        # Load the models using slide2vec in a sub process so that when each model is done, the VRAM gets properly cleared.
        subprocess.run(
            [sys.executable, "run_single_model.py", model_name],
            env=env,
            check=True,
        )

if __name__ == "__main__":
    main()
