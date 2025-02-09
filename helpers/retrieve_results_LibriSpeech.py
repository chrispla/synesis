import json
import sys
from itertools import product
from pathlib import Path

import wandb

# set cwd as the root of the project
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from synesis.utils import get_metric_from_wandb


def get_equi_run_names():
    return [
        f"2_{eval_type}_{task}_{transform}_{label}_{ds}_{feature}"
        for eval_type, task, transform, label, ds, feature in product(
            equi_types, equi_tasks, transforms, equi_labels, dataset, features
        )
    ]


def get_info_run_names():
    return [
        f"2_{eval_type}_{task}_{ds}_{label}_{feature}"
        for eval_type, task, ds, label, feature in product(
            info_types, info_tasks, dataset, info_labels, features
        )
    ]


all_results = {}

entity = "cplachouras"
project = "synesis"

dataset = ["LibriSpeech"]
equi_types = ["EQUI_PARA"]
info_types = ["INFO_DOWN"]
features = [
    "MDuo",
    "AudioMAE",
    "Wav2Vec2",
    "HuBERT",
    "CLAP",
    "Whisper",
    "UniSpeech",
    "XVector",
]
info_tasks = ["regression", "regression_linear"]
equi_tasks = ["regression", "regression_linear"]
equi_labels = ["dummy"]
info_labels = ["wps"]
transforms = ["PitchShift", "AddWhiteNoise", "TimeStretch"]

wandb_runs = wandb.Api().runs(f"{entity}/{project}")
run_names = get_info_run_names() + get_equi_run_names()

# retrieve run_id from each run that matches local and wandb
run_ids = {}
for run_name in run_names:
    for run in wandb_runs:
        if run_name == run.name:
            run_ids[run_name] = run.id
            break

# construct wandb paths to later be used with
# entity, project, run_id, model_name = model.split("/")
wandb_paths = []
for run_name, run_id in run_ids.items():
    wandb_paths.append(f"{entity}/{project}/{run_id}/{run_name}")

# Track success/failure
successful_runs = []
failed_runs = []

# evaluate
for i, wandb_path in enumerate(wandb_paths):
    try:
        entity, project, run_id, model_name = wandb_path.split("/")
        run = wandb.Api().run(f"{entity}/{project}/{run_id}")

        results = get_metric_from_wandb(run, "MSE")
        if not results:
            results = get_metric_from_wandb(run, "mse")
        if not results:
            results = get_metric_from_wandb(run, "Mean Squared Error")
        if not results:
            failed_runs.append((run.name, "No MSE metric found"))
            print(f"✗ Failed: {run.name}")
            print("  > Error: No MSE metric found")
        else:
            successful_runs.append(run.name)
            print(f"✓ Success: {run.name}")
            all_results[run.name] = {"mse": results}
    except Exception as e:
        failed_runs.append((run.name, str(e)))
        print(f"✗ Failed: {run.name}")
        print(f"  > Error: {str(e)}")

    print("> PROGRESS: {}/{}".format(i + 1, len(wandb_paths)))

# Print summary
print("\n=== Evaluation Summary ===")
print(f"Total runs: {len(wandb_paths)}")
print(f"Successful: {len(successful_runs)}")
print(f"Failed: {len(failed_runs)}")

# Save results
results_dir = Path("results")
if not results_dir.exists():
    results_dir.mkdir()
# Save json
results_json = results_dir / "LibriSpeech_results_v2.json"
with open(results_json, "w") as f:
    json.dump(all_results, f, indent=4, sort_keys=True)
