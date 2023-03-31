# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import re
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results-dir",
    "-r",
    type=str,
    default="./results",
    help="directory containing sweep results",
)
parser.add_argument("--version", "-v", type=int, default=0, help="version to be collected")
parser.add_argument("--gen-text-n", "-n", type=int, default=1, help="expected number of generated text")
parser.add_argument("--output", "-o", type=str, default="./results.csv", help="output file")
args = parser.parse_args()


def get_branch(file_path):
    match = re.match(r".*\/(.*)\.log", file_path)
    if match is None:
        return False
    else:
        return match.groups()[0]


def get_benchmark_params(root_dir, file_path):
    match = re.match(
        rf"{root_dir}\/(.+?)_(fp\d+)_(true|false)_(true|false)_(\d+)gpus_v(\d+)\/",
        file_path,
    )
    if match is None:
        return False
    else:
        model, dtype, graphs, kernel, gpus, version = match.groups()
        bool_dict = {"true": True, "false": False}
        return {
            "model": model,
            "dtype": dtype,
            "graphs": bool_dict[graphs.lower()],
            "kernel": bool_dict[kernel.lower()],
            "gpus": int(gpus),
            "version": int(version),
        }


def get_perf_data(file_content):
    matches = re.findall(r"\s+(.+?)\sLatency:\s+(\d+\.\d+)\sms", file_content)
    if matches is []:
        return False
    else:
        return {f"latency-{key}": float(val) for key, val in matches}


def get_generated_text(file_content, gen_text_n):
    file_content = file_content.replace("\n", " ")
    file_content = file_content.replace("\t", " ")
    matches = re.findall(r"RESPONSE\s(\d+):\s+[-]{30}\s+(.+?)\s+[-]{30}", file_content)
    if len(matches) != gen_text_n:
        return False
    else:
        return {f"generated-text-{key}": val for key, val in matches}


def get_error(file_content):
    matches = re.findall(r"Error:\s+(.+?)\n", file_content)
    if matches is []:
        return False
    else:
        return {f"error": val for val in matches}


if __name__ == "__main__":
    # List to collect data from all benchmarks
    benchmarks_data = []

    # Walk through directory of results from sweep.sh
    for root, dirs, files in os.walk(args.results_dir):
        # Because of how some models are named, the dir structure for results can vary, e.g.:
        # "EleutherAI/gpt-neo_*/baseline.log" versus "gpt2_*/baseline.log"
        if dirs:
            continue

        # Get data from baseline and each tested branch
        for name in files:
            file_path = os.path.join(root, name)

            branch = get_branch(file_path)
            if not branch:
                print(f"WARNING: Could not detect branch for file {file_path}, skipping")
                continue

            params = get_benchmark_params(args.results_dir, file_path)
            if not params:
                print(f"WARNING: Could not detect benchmark settings for file {file_path}, skipping")
                continue

            # Verify that the version matches that which we want to collect
            if params["version"] != args.version:
                continue

            with open(file_path, "r") as f:
                file_content = f.read()

            perf_data = get_perf_data(file_content)
            if not perf_data:
                print(f"WARNING: Could not detect benchmark performance data for file {file_path}")

            generated_text = get_generated_text(file_content, args.gen_text_n)
            if not generated_text:
                print(f"WARNING: Could not detect generated text for file {file_path}")

            error = get_error(file_content)
            if error:
                print(f"Error found in {file_path}, collecting error info...")
                benchmarks_data.append({"branch": branch, **params, **error})
                continue

            benchmarks_data.append({"branch": branch, **params, **perf_data, **generated_text})

    # Convert to a DataFrame and save
    benchmarks_df = pd.DataFrame(benchmarks_data)
    benchmarks_df.to_csv(args.output)
