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

experiments = [
    {
        "name":"preprocess_no-preprcs",
        "data_params"={
            "preprocess_desc":"no-preprcs",
        },
        "embedding_params":{
            "num_epochs":1
        },
        "selectivity_prams":{
            "num_epochs":1
        }
    },
    {
        "name":"preprocess_2024preprcs",
        "data_params"={
            "preprocess_desc":"2024preprcs",
        },
        "embedding_params":{
            "num_epochs":1
        },
        "selectivity_prams":{
            "num_epochs":1
        }
    }
]

training_seeds = [1234, 2345]

split_seed = 1234

experiment_name = "preprcs_ex01"


base_output_path = f"outputs/{experiment_name}/"

def main():
    os.makedirs(base_output_path, exist_ok=True)

    all_results = []

    total_runs = len(experiments) * len(training_seeds)
    run_id = 0

    for experiment in experiments:
        
        print("\n"+"="*60)
        print(f"Experiment:{experiment["name"]}")
        print("="*60)

        for training_seed in training_seeds:
            run_id += 1

            print(f"\nRun {run_id}/{total_run}")
            print(f"Config: {experiment["name"]}")
            print(f"Training seed: {training_seed}")

            output_path = (
                base_output_path + f"run_{run_id:03d}/"
            )

            os.makedirs(output_path, exist_ok=True)

            results = run_experiment(
                training_seed=training_seed,
                split_seed=split_seed,
                output_path=output_path,
                data_params=experiment["data_params"],
                embedding_prams=experiment["embedding_params"],
                selectivity_params=experiment["selectivity_params"]
            )

            result_row = {
                "run_id": run_id,
                "experiment_name": experiment["name"],
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

            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(
                base_output_path + "summary.csv",
                index=False
            )

            print("\nResults:")
            print(results)
    
    print("\nAll experiments completed.")
    print(f"Result saved to {base_output_path}summary.csv")

if __name__ == "__main__":
    main()
