from pathlib import Path
import os, shutil, subprocess, sys


WORK_DIR = Path(sys.argv[1])
REMOTE_DIR = Path(sys.argv[2])
OUT_DIR = WORK_DIR/'features'
BATCHS_SIZE = 5
BATCH_DIR = WORK_DIR/'dataset'
MODELS = ['virchow2', 'gigapath', 'conchv15', 'hoptimus-1']
TOKENS = {}


def fetch_processed():
    processed = {}
    for model_name in MODELS:
        for file in (OUT_DIR/model_name).glob('*.pt'): #Slide2vec outputs under OUT_DIR/model_name/*.pt
            if processed.get(file.stem) is None:
                processed[file.stem] = 1
            else:
                processed[file.stem] += 1
    return processed


def fetch_batch(processed:dict):
    current_batch = []
    for folder in REMOTE_DIR.iterdir():
        for file in (folder/'kidney').glob("*.svs"):
            file_count = 0 if (value := processed.get(file.stem)) is None else value
            if file_count < 4: # Not all models have output this file
                current_batch.append(file)
                if len(current_batch) >= BATCHS_SIZE:
                    return current_batch

    return current_batch




def main():
    while True:
        processed = fetch_processed()
        current_batch = fetch_batch(processed)
        if not current_batch: # We're done!
            print("WORKS DONE")
            quit(101)

        # Download the current batch
        BATCH_DIR.mkdir(parents=True, exist_ok=True)
        for file in current_batch:
            shutil.copy2(file, BATCH_DIR / file.name)


        # Fetch HF token
        with open("TOKENS.csv", 'r') as file:
            for line in file:
                line = line.split(',')
                TOKENS[line[0]] = line[1].rstrip('\n')

        for model in MODELS:
            env = os.environ.copy()
            env["HF_TOKEN"] = TOKENS[model]

            # Start feature extraction as a subprocess (that way vram gets cleared properly when the code is done)
            subprocess.run(
                [sys.executable, "feature_extraction.py", model, BATCH_DIR.resolve(), (OUT_DIR/model).resolve()],
                env=env,
                check=True,
            )

        shutil.rmtree(BATCH_DIR)

if __name__ == "__main__":
    main()
