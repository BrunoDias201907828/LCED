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
    params_file = "modeling/parameters_svr.json" 

    run_script(script_path, "svm", "NoImputation", "TargetEncoding", params_file, "svm_t", "svm_t")
    run_script(script_path, "svm", "BayesianRidge", "TargetEncoding", params_file, "svm_tb", "svm_tb")
    run_script(script_path, "svm", "NoImputation", "BinaryEncoding", params_file, "svm_b", "svm_b")
    run_script(script_path, "svm", "BayesianRidge", "BinaryEncoding", params_file, "svm_bb", "svm_bb")

#python3 modeling/train_script.py --model svm --imputation NoImputation --encoding TargetEncoding --params '{"kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"], "C": [1, 10, 100, 1000], "epsilon": [0.01, 0.1, 0.5]}' --run_name svm_t --experiment_name svm_t &
