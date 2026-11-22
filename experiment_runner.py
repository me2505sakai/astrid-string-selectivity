import os
import pandas as pd

from AstridEmbed import run_experiment

# results = run_experiment(
#     training_seed=1234,
#     split_seed=1234,
#     output_path="outputs/00/",
#     data_params={
#         "dataset_path":"datasets/dblp/",
#         "file_name_prefix":"dblp_titles",
#         "query_type":"unique-substring",
#         "preprocess_desc":"no-preprcs",
#     }
#     embedding_params={
#         "embedding_dimension":64,
#         "batch_size":128,
#         "num_epochs":32,
#         "margin":0.2,
#         "lr":0.001,
#         "channel_size":8,
#         "log_by_batch":True
#     },
#     selectivity_params={
#         "embedding_dimension":64,
#         "batch_size":128,
#         "num_epochs":64,
#         "lr":0.001,
#         "min_val":0.0,
#         "max_val":1.0,
#         "log_by_batch":True
#     }
# )

# paramsは{}とするとデフォルト値(上記)が使われる
experiments = [
    {
        "name":"preprocess_no-preprcs",
        "data_params":{
            "preprocess_desc":"no-preprcs",
        },
        "embedding_params":{
            "num_epochs":1
        },
        "selectivity_params":{
            "num_epochs":1
        }
    },
    {
        "name":"preprocess_2024preprcs",
        "data_params":{
            "preprocess_desc":"2024preprcs",
        },
        "embedding_params":{
            "num_epochs":1
        },
        "selectivity_params":{
            "num_epochs":1
        }
    }
]

training_seeds = [1234, 2345]

split_seeds = [1234, 2345]

experiment_name = "develop-test2"


base_output_path = os.path.join("outputs",experiment_name)
summary_csv_path = os.path.join(base_output_path, "summary.csv")

def main():
    os.makedirs(base_output_path, exist_ok=True)

    print(f"Starting experiment in ({base_output_path}).")

    all_results = None
    completed_keys = None
    run_id = None

    #すでにsummary.csvがある場合、完了済み条件をスキップ
    if os.path.exists(summary_csv_path):
        existing_df = pd.read_csv(summary_csv_path)
        all_results = existing_df.to_dict(orient="records")
        completed_keys = set(
            zip(
                existing_df["experiment_name"],
                existing_df["split_seed"],
                existing_df["training_seed"],
            )
        )
        run_id = int(existing_df["run_id"].max())
        print(f"Resuming from existing summary.csv ({len(completed_keys)} runs already completed).")
    else:
        all_results = []
        completed_keys = set()
        run_id = 0

    total_runs = len(experiments) * len(split_seeds) * len(training_seeds)
    current_idx = 0

    for experiment in experiments:
        exp_name = experiment["name"]
        print("\n"+"="*60)
        print(f"Experiment:{exp_name}")
        print("="*60)

        for split_seed in split_seeds:
            for training_seed in training_seeds:
                current_idx += 1

                if (exp_name, split_seed, training_seed) in completed_keys:
                    print(
                        f"[Skip {current_idx}/{total_runs}] "
                        f"{exp_name} | split={split_seed}, train={training_seed}"
                    )
                    continue
                
                run_id += 1
                print(f"\nRun {run_id} (Progress: {current_idx}/{total_runs})")
                print(f"Config: {exp_name} | split_seed: {split_seed} | training_seed: {training_seed}")

                output_path = os.path.join(base_output_path, f"run_{run_id:03d}")
                os.makedirs(output_path, exist_ok=True)

                results = run_experiment(
                    training_seed=training_seed,
                    split_seed=split_seed,
                    output_path=output_path + "/",
                    data_params=experiment["data_params"],
                    embedding_params=experiment["embedding_params"],
                    selectivity_params=experiment["selectivity_params"]
                )

                result_row = {
                    "run_id": run_id,
                    "experiment_name": exp_name,
                    "training_seed": training_seed,
                    "split_seed": split_seed,
                    **experiment["data_params"],
                    **{
                        f"embedding_{k}":v
                        for k,v in experiment["embedding_params"].items()
                    },
                    **{
                        f"selectivity_{k}":v
                        for k,v in experiment["selectivity_params"].items()
                    },
                    **results
                }

                all_results.append(result_row)
                completed_keys.add((exp_name, split_seed, training_seed))

                summary_df = pd.DataFrame(all_results)
                summary_df.to_csv(summary_csv_path, index=False)

                print("Results:",results)
    
    print("\nAll experiments completed.")
    print(f"Result saved to {base_output_path}summary.csv")

if __name__ == "__main__":
    main()
