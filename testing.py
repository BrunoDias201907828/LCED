import subprocess
import json

def write_to_file(script_path, filename, return_code):
    with open(filename, "a") as f:
        f.write(f"Finished running {script_path} with return code {return_code}\n")

def error_to_file(script_path, e, filename):
    with open(filename, "a") as f:
        f.write(f"Error running {script_path}: {e}\n")


def run_script(script_path, model, imputation, encoding, params_file, run_name, experiment_name, filename="log.txt"):
    with open(params_file, "r") as f:
        params = json.load(f)

    command = [
        "python3",
        script_path,
        "--model", model,
        "--imputation", imputation,
        "--encoding", encoding,
        "--params", json.dumps(params),
        "--run_name", run_name,
        "--experiment_name", experiment_name,
    ]

    completed_process = subprocess.run(command, capture_output=True, text=True)
    if completed_process.returncode == 0:
        write_to_file(script_path, filename, completed_process.returncode)
    else:
        error_to_file(script_path, completed_process.stderr, filename)
    return True

if __name__ == "__main__":
    script_path = "modeling/train_script.py"  
    params_file = "params.json" 

    run_script(script_path, "xgboost", "NoImputation", "TargetEncoding", params_file, "xg_t", "xg_t")
    run_script(script_path, "xgboost", "BayesianRidge", "TargetEncoding", params_file, "xg_tb", "xg_tb")
    run_script(script_path, "xgboost", "NoImputation", "BinaryEncoding", params_file, "xg_b", "xg_b")
    run_script(script_path, "xgboost", "BayesianRidge", "BinaryEncoding", params_file, "xg_bb", "xg_bb")

