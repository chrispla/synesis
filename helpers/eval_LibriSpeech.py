import json
from itertools import product
from pathlib import Path

import wandb
from synesis.equivariance.features import evaluate as EQUI_FEAT_eval
from synesis.equivariance.parameters import evaluate as EQUI_PARA_eval
from synesis.informativeness.downstream import evaluate as INFO_DOWN_eval


def get_equi_run_names():
    return [
        f"{eval_type}_{task}_{transform}_{label}_{ds}_{feature}"
        for eval_type, task, transform, label, ds, feature in product(
            equi_types, equi_tasks, transforms, equi_labels, dataset, features
        )
    ]


def get_info_run_names():
    return [
        f"{eval_type}_{task}_{ds}_{label}_{feature}"
        for eval_type, task, ds, label, feature in product(
            info_types, info_tasks, dataset, info_labels, features
        )
    ]


all_results = {}

entity = "cplachouras"
project = "synesis"

dataset = ["LibriSpeech"]
equi_types = ["EQUI_PARA", "EQUI_FEAT"]
info_types = ["INFO_DOWN"]
features = [
    "MDUO",
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
run_names = get_equi_run_names() + get_info_run_names()

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

        # if run.summary["evaluation_metrics"] exists, the run has already been evaluated
        # try:
        #     if "evaluation_metrics" in run.summary:
        #         print(f"✓ Already evaluated: {run.name}")
        #         successful_runs.append(run.name)
        #         continue
        # except:
        #     pass

        artifact_base_name = run.logged_artifacts()[0].name

        print(f"\nEvaluating run: {run.name}")

        if "INFO_DOWN" in run.name:
            if "hue" in wandb_path:
                label = "hue"
            elif "saturation" in wandb_path:
                label = "saturation"
            else:
                label = "brightness"
            results = INFO_DOWN_eval(
                model=f"{entity}/{project}/{run_id}/{artifact_base_name}",
                feature=run.config["feature"],
                dataset=run.config["dataset"],
                task=run.config["task"],
                task_config={"evaluation": {"batch_size": 1}},
                label=label,
                item_format="raw",
                device="cuda",
                logging=True,
            )

        if "EQUI_PARA" in run.name:
            if "HueShift" in wandb_path:
                transform = "HueShift"
                label = "hue"
            elif "BrightnessShift" in wandb_path:
                transform = "BrightnessShift"
                label = "brightness"
            else:
                transform = "SaturationShift"
                label = "saturation"
            results = EQUI_PARA_eval(
                model=f"{entity}/{project}/{run_id}/{artifact_base_name}",
                feature=run.config["feature"],
                dataset=run.config["dataset"],
                transform=transform,
                task_config={"evaluation": {"batch_size": 1}},
                label=label,
                task=run.config["task"],
                device="cuda",
                logging=True,
            )

        if "EQUI_FEAT" in run.name:
            if "HueShift" in wandb_path:
                transform = "HueShift"
                label = "hue"
            elif "BrightnessShift" in wandb_path:
                transform = "BrightnessShift"
                label = "brightness"
            else:
                transform = "SaturationShift"
                label = "saturation"
            results = EQUI_FEAT_eval(
                model=f"{entity}/{project}/{run_id}/{artifact_base_name}",
                feature=run.config["feature"],
                dataset=run.config["dataset"],
                transform=transform,
                label=label,
                task=run.config["task"],
                task_config={"evaluation": {"batch_size": 1}},
                device="cuda",
                logging=True,
            )

        successful_runs.append(run.name)
        print(f"✓ Success: {run.name}")
        all_results[run.name] = results

    except Exception as e:
        failed_runs.append((run.name, str(e)))
        print(f"✗ Failed: {run.name}")
        print(f"  > Error: {str(e)}")
        continue

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
results_json = results_dir / "LibriSpeech_results.json"
with open(results_json, "w") as f:
    json.dump(all_results, f, indent=4, sort_keys=True)
