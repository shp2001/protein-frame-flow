import os
import subprocess
import re
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_dockq(native_pdb, model_pdb, dockq_path="./DockQ.py"):
    cmd = ["python", dockq_path, model_pdb, native_pdb]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr.strip()}

    output = result.stdout
    scores = {}
    for key in ["fnat", "iRMSD", "LRMSD", "DockQ"]:
        match = re.search(rf"{key}:\s*([\d\.]+)", output)
        if match:
            scores[key] = float(match.group(1))
    return scores

def process_one(native_pdb, model_sample_path, dockq_path, pdb_id, sample):
    print(f"Processing: {pdb_id}_{sample}")
    score = run_dockq(native_pdb, model_sample_path, dockq_path)
    output_json = os.path.join(os.path.dirname(model_sample_path), "dockq_score.json")
    with open(output_json, "w") as f:
        json.dump(score, f, indent=2)
    return (pdb_id, sample, score)

def batch_run(native_pdb_dir='/home/psh/benchmark_after210930/pdb', 
              pred_dir='/home/psh/af3_output_processed',
              dockq_path="/home/psh/DockQ/src/DockQ/DockQ.py",
              max_workers=4):

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for fname in sorted(os.listdir(native_pdb_dir)):
            if not fname.endswith(".pdb"):
                continue
            native_pdb = os.path.join(native_pdb_dir, fname)
            pdb_id = fname.replace('.pdb', '')

            model_path = os.path.join(pred_dir, pdb_id)
            if not os.path.isdir(model_path):
                continue

            for sample in os.listdir(model_path):
                model_sample_path = os.path.join(model_path, sample, f"{pdb_id}_filtered.pdb")
                if not os.path.isfile(model_sample_path):
                    continue
                futures.append(
                    executor.submit(process_one, native_pdb, model_sample_path, dockq_path, pdb_id, sample)
                )

        for future in as_completed(futures):
            pdb_id, sample, score = future.result()
            print(f"Finished {pdb_id}_{sample}: {score}")

if __name__ == "__main__":
    native_pdb_dir = '/home/psh/benchmark_after210930/pdb'             
    pred_dir = '/home/psh/af3_output_processed'    
    dockq_path = "/home/psh/DockQ/src/DockQ/DockQ.py"             

    batch_run(native_pdb_dir, pred_dir, dockq_path, max_workers=20)
