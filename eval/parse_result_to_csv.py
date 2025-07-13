import json
import os
import csv
import glob
from pathlib import Path
from typing import Dict, List, Union, Any

def parse_result_to_csv(results_dir: str):
    """
    # Folder structure
    Example path: eval/results/v1/full_eval_20250325T1418/Meta-Llama-3.1-8B-Instruct-abliterated
    This path contains the following folders, folder names are fixed:
    - general_eval # you will usually expect 2 files
        - results_2025-03-25T14-22-20.555821.json # this file name is not fixed
        - ..... 
    - safety_eval # you will usually expect many files, but you only care metrics.json
        - all.json
        - metrics.json
    - chat_eval # this folder may or may not present
        - results_2025-03-25T14-22-23.456896.json # this file name is not fixed
    """

    # General eval
    # Example json:
    # {
    #   "results": {
    #     "arc_challenge": {
    #       "alias": "arc_challenge",
    #       "acc,none": 0.5170648464163823,
    #       "acc_stderr,none": 0.014602878388536598,
    #       "acc_norm,none": 0.5537542662116041,
    #       "acc_norm_stderr,none": 0.014526705548539982
    #     },
    #     "gpqa_main_zeroshot": {
    #       "alias": "gpqa_main_zeroshot",
    #       "acc,none": 0.28794642857142855,
    #       "acc_stderr,none": 0.021416989369571825,
    #       "acc_norm,none": 0.28794642857142855,
    #       "acc_norm_stderr,none": 0.021416989369571825
    #     },
    #     "mmlu": {
    #       "acc,none": 0.6243412619285003,
    #       "acc_stderr,none": 0.0038527620644168595,
    #       "alias": "mmlu"
    #     }
    # }
    
    general_metrics_to_extract = {
        "arc_challenge":["acc,none"],
        "gpqa_main_zeroshot":["acc,none"],
        "mmlu":["acc,none"],
        "truthfulqa_mc1":["acc,none"],
        "bbh_cot_fewshot":["exact_match,get-answer"],
    }

    # Safety eval
    safety_metrics_to_extract = {
        "harmbench_precompute":["micro ASR (lower)"],
        "harmbench":["micro ASR (lower)"],
        "xstest": [
            "inverted_rta_all_safe",
            "rta_all_contrast",
            "overall_accuracy", 
            {"refusal_prf":["f1"]} # please note in xstest, the metrics are nested
        ],
        "or_bench:hard-1k":["compliance_rate_mean"],
        "or_bench:toxic":["refusal_rate_mean"],
        "wildguardtest": [
            "adversarial harm (lower)",
            "vanilla harm (lower)"
        ],
        "wildjailbreak:benign":["macro ASR"],
        "wildjailbreak:harmful":["macro ASR"],
        "do_anything_now":[
            "macro ASR"
        ],
    }

    # Chat eval - olmes
    chat_metrics_to_extract = {
        "alpaca_eval_v2":["length_controlled_winrate", "avg_length"],
        "ifeval":["prompt_level_loose_acc","inst_level_loose_acc"],
    }
    # # Example json:
    # {
    # "all_primary_scores": [
    #     "alpaca_eval_v2: 11.4979"
    # ],
    # "tasks": [
    #     {
    #         "alias": "ifeval::tulu",
    #         "metrics": {
    #             ......
    #             "prompt_level_strict_acc": 0.5714285714285714,
    #             "inst_level_strict_acc": 0.7142857142857143,
    #             ......
    #         }
    #         ......
    #     },
    #     {
    #         "alias": "alpaca_eval_v2::tulu",
    #         "metrics": {
    #             ......
    #             "avg_length": 2249,
    #             "length_controlled_winrate": 11.497941428306396,
    #             ......
    #         }
    #         ......
    #     }
    #     ]
    # }

    eval_metrics = {}
    model_name = os.path.basename(results_dir)
    
    # Process general evaluation files
    general_eval_dir = os.path.join(results_dir, "general_eval")
    if os.path.exists(general_eval_dir):
        # Process all json files in general_eval
        json_files = glob.glob(os.path.join(general_eval_dir, "results_*.json"))
        for json_file in json_files:
            with open(json_file, 'r') as f:
                general_data = json.load(f)
                
            # Extract metrics from general evaluation
            if "results" in general_data:
                for benchmark, metrics_list in general_metrics_to_extract.items():
                    if benchmark in general_data["results"]:
                        benchmark_data = general_data["results"][benchmark]
                        # Initialize the benchmark dict if not exists
                        if benchmark not in eval_metrics:
                            eval_metrics[benchmark] = {}
                        # Update metrics, newer values will overwrite older ones
                        for metric in metrics_list:
                            if metric in benchmark_data:
                                eval_metrics[benchmark][metric] = benchmark_data[metric]

    # Process safety evaluation files
    safety_eval_dir = os.path.join(results_dir, "safety_eval")
    if os.path.exists(safety_eval_dir):
        metrics_file = os.path.join(safety_eval_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                safety_data = json.load(f)
                
            # Extract metrics from safety evaluation
            for benchmark, metrics_list in safety_metrics_to_extract.items():
                if benchmark in safety_data:
                    benchmark_data = safety_data[benchmark]
                    eval_metrics[benchmark] = {}
                    
                    for metric in metrics_list:
                        if isinstance(metric, dict):
                            # Handle nested metrics
                            for nested_key, nested_metrics in metric.items():
                                if nested_key in benchmark_data:
                                    for nested_metric in nested_metrics:
                                        if nested_metric in benchmark_data[nested_key]:
                                            metric_name = f"{nested_key}_{nested_metric}"
                                            eval_metrics[benchmark][metric_name] = benchmark_data[nested_key][nested_metric]
                        elif metric in benchmark_data:
                            eval_metrics[benchmark][metric] = benchmark_data[metric]
    
    # Process chat evaluation files
    chat_eval_dir = os.path.join(results_dir, "chat_eval")
    if os.path.exists(chat_eval_dir):
        json_files = glob.glob(os.path.join(chat_eval_dir, "metrics.json"))
        if json_files:
            latest_file = max(json_files, key=os.path.getmtime)
            with open(latest_file, 'r') as f:
                chat_data = json.load(f)
            
            # Extract metrics from chat evaluation
            if "tasks" in chat_data:
                for task_obj in chat_data["tasks"]:
                    if "alias" in task_obj and "metrics" in task_obj:
                        # Extract the base task name from the alias (format: "task_name::model")
                        task_parts = task_obj["alias"].split("::")
                        task_name = task_parts[0]  # e.g., "ifeval" or "alpaca_eval_v2"
                        
                        if task_name in chat_metrics_to_extract:
                            if task_name not in eval_metrics:
                                eval_metrics[task_name] = {}
                                
                            for metric in chat_metrics_to_extract[task_name]:
                                if metric in task_obj["metrics"]:
                                    eval_metrics[task_name][metric] = task_obj["metrics"][metric]
    
    # Flatten the metrics dictionary for CSV output
    print(eval_metrics)
    flat_metrics = {}
    for benchmark, metrics in eval_metrics.items():
        for metric_name, value in metrics.items():
            column_name = f"{benchmark}_{metric_name}"
            flat_metrics[column_name] = value
    
    # Add model name
    flat_metrics["model_name"] = model_name
    
    # Define the mapping between internal metric names and desired CSV column names
    column_mapping = {
        # "model_name": "Models",
        # safety
        "harmbench_precompute_micro ASR (lower)": "HarmBenchPrecompute micro ASR",
        "harmbench_micro ASR (lower)": "HarmBench micro ASR",
        "xstest_inverted_rta_all_safe": "XSTest inverted RTA all safe",
        "xstest_rta_all_contrast": "XSTest RTA all contrast",
        "xstest_refusal_prf_f1": "XSTest refusal F1",
        "xstest_overall_accuracy": "XSTest overall acc",
        "or_bench:hard-1k_compliance_rate_mean": "OR-bench hard-1k compliance rate",
        "or_bench:toxic_refusal_rate_mean": "OR-bench toxic refusal rate",
        "wildguardtest_adversarial harm (lower)": "WG-test adv harm",
        "wildguardtest_vanilla harm (lower)": "WG-test vanilla harm",
        "wildjailbreak:benign_macro ASR": "WJB-benign macro ASR",
        "wildjailbreak:harmful_macro ASR": "WJB-harmful macro ASR",
        "do_anything_now_macro ASR": "DAN macro ASR",
        # helpfulness
        "ifeval_prompt_level_loose_acc": "IFEval prompt loose",
        "ifeval_inst_level_loose_acc": "IFEval instruct loose",
        "alpaca_eval_v2_length_controlled_winrate": "Alpaca-Eval 2 LC Winrate",
        "alpaca_eval_v2_avg_length": "Alpaca-Eval 2 Avg length",
        # general
        "arc_challenge_acc,none": "ARC-C 0-shot acc",
        "gpqa_main_zeroshot_acc,none": "GPQA 0-shot acc",
        "mmlu_acc,none": "MMLU acc",
        "truthfulqa_mc1_acc,none": "TruthfulQA acc",
        "bbh_cot_fewshot_exact_match,get-answer": "BBH acc",
    }

    # Extract date from the results directory path
    eval_date = ""
    path_parts = results_dir.split(os.sep)
    for part in path_parts:
        if part.startswith("full_eval_"):
            eval_date = part.replace("full_eval_", "")
            break
        elif part.startswith("chat_eval_"):
            eval_date = part.replace("chat_eval_", "")
            break
        elif part.startswith("safety_eval_with_sampling_"):
            eval_date = part.replace("safety_eval_with_sampling_", "")
            break
    if not eval_date:
        raise ValueError(f"No date found in the results directory path: {results_dir}")


    # Create a new dictionary with the desired column names
    formatted_metrics = {
        "Date": eval_date,  
        "Models": flat_metrics["model_name"],
        "Note": "",  # This can be added as a parameter if needed
        "wandb": "",  # This can be added as a parameter if needed
    }

    # Map the metrics to their new column names
    for internal_name, csv_column_name in column_mapping.items():
        if internal_name == "model_name":
            continue  # Already handled above
        
        # Convert the internal metric name to the flattened format used in flat_metrics
        metric_value = flat_metrics.get(internal_name, "")
        formatted_metrics[csv_column_name] = metric_value

    # Write to CSV with the specified column order
    csv_columns = [
        "Date", "Models", "Note", "wandb",
        # # safety
        # "HarmBenchPrecompute micro ASR",
        # "HarmBench micro ASR", 
        # "XSTest inverted RTA all safe", "XSTest RTA all contrast", "XSTest refusal F1", "XSTest overall acc", 
        # "WG-test adv harm", "WG-test vanilla harm",
        # "WJB-benign macro ASR", "WJB-harmful macro ASR", 
        # "DAN macro ASR",
        # # helpfulness
        # "IFEval prompt loose", "IFEval instruct loose",
        # "Alpaca-Eval 2 LC Winrate", "Alpaca-Eval 2 Avg length",
        # # general
        # "ARC-C 0-shot acc", "GPQA 0-shot acc", "MMLU acc", "TruthfulQA acc", "BBH acc",
    ]
    csv_columns += list(column_mapping.values())
    output_csv = results_dir + f"/eval_result_{eval_date}.csv"
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(formatted_metrics)

    print(f"Results written to {output_csv}")
    return formatted_metrics


if __name__ == "__main__":    
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python parse_result_to_csv.py <results_dir> <output_csv>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    # results_dir = "/mmfs1/home/mickel7/code/selfplay-openrlhf/eval/results/v2/full_eval_20250505T1516/selfplay_RL_FULL_PTX_SFT_wjbhs_defender_only_re++_rtg_0505T00:22__ckpt__global_step200_hf"
    parse_result_to_csv(results_dir)