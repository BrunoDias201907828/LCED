import subprocess
import json

def write_to_file(script_path, filename, return_code):
    with open(filename, "a") as f:
        f.write(f"Finished running {script_path} with return code {return_code}\n")

def error_to_file(script_path, e, filename):
    with open(filename, "a") as f:
        f.write(f"Error running {script_path}: {e}\n")


def run_script(
        script_path: str, model: str, imputation: bool, external: bool, encoding: str, params_file: str, run_name: str,
        experiment_name: str, filename="log.txt"
):

    command = [
        "python3",
        script_path,
        "--model", model,
        "--encoding", encoding,
        "--params_path", params_file,
        "--run", run_name,
        "--experiment", experiment_name,
    ]
    if imputation:
        command += ["--imputation"]
    if external:
        command += ["--external"]

    completed_process = subprocess.run(command, capture_output=True, text=True)
    if completed_process.returncode == 0:
        write_to_file(script_path, filename, completed_process.returncode)
    else:
        error_to_file(script_path, completed_process.stderr, filename)
    return True


if __name__ == "__main__":
    experiment = "SingleRegressor"
    script_path = "modeling/train_script.py"
    models = ["linear_regression", "elastic_net", "decision_tree", "bayesian_ridge", "sgd"]

    # Linear Regression
    for model in models:
        model_name_cap = model.replace("_", " ").title().replace(" ", "")
        params_file = f"modeling/parameters_{model}.json"
        for encoding in ["TargetEncoding", "BinaryEncoding"]:
            run_script(script_path, model, False, False, encoding,
                       params_file, f"{model_name_cap}_{encoding}", experiment)
            run_script(script_path, model, True, False, encoding,
                       params_file, f"{model_name_cap}_{encoding}_Imputation", experiment)
            run_script(script_path, model, False, True, encoding,
                       params_file, f"{model_name_cap}_{encoding}_External", experiment)
            run_script(script_path, model, True, True, encoding,
                       params_file, f"{model_name_cap}_{encoding}_Imputation_External", experiment)
