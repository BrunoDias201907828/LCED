import scikit_posthocs as sp
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
import mlflow
import os


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow_infos = [
        [
            "rf_t", "rf_b", "rf_tb", "rf_bb", "xg_t", "xg_b", "xg_tb", "xg_bb",
            "lsvr_t", "lsvr_b", "lsvr_tb", "lsvr_bb", "linear"
        ],
        [
            "linear", "lsvr_t", "rf_t", "xg_bb"
        ]
    ]
    for mlflow_info, suffix in zip(mlflow_infos, ["", "compact"]):
        data = []
        i = 0
        for run_name in mlflow_info:
            i += 1
            print(i / len(mlflow_info))
            mlflow.set_experiment(experiment_name=run_name)
            runs = mlflow.search_runs()
            runs = runs[runs['tags.mlflow.runName'] == run_name]
            run_id = runs.iloc[0].run_id
            with tempfile.TemporaryDirectory() as temp_dir:
                mlflow.artifacts.download_artifacts(artifact_path="cv_results.csv", dst_path=temp_dir, run_id=run_id)
                downloaded_csv_path = os.path.join(temp_dir, "cv_results.csv")
                df = pd.read_csv(downloaded_csv_path)
                _df = df.loc[
                    (df.iter == df.iter.max()) &
                    (df.mean_test_score == df.loc[df.iter == df.iter.max()].mean_test_score.max())
                ]
                columns = [f"split{i}_test_score" for i in range(5)]
                _data = [(i, run_name if run_name != "linear" else "BASELINE", _df[col].values[0]) for i, col in enumerate(columns)]
                data.extend(_data)
        df = pd.DataFrame(data, columns=["cv_fold", "estimator", "score"])
        avg_rank = df.groupby('cv_fold').score.rank().groupby(df.estimator).mean()
        print("Rank Done")
        test_results = sp.posthoc_conover_friedman(
            df,
            melted=True,
            block_col='cv_fold',
            group_col='estimator',
            y_col='score',
        )
        print("Posthoc Done")
        plt.figure(figsize=(16, 8), dpi=100)
        plt.title('Critical difference diagram of average score ranks')
        sp.critical_difference_diagram(avg_rank, test_results)
        plt.savefig(f'critical_diff_diagram_{suffix}.png')
