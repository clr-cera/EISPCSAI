import os
from codecarbon import track_emissions
from download_models import download_models
from extract_features import extract_dino_features, extract_features
from visualizations import generate_visualizations
from eval_ensemble import (
    eval_ensemble,
    eval_ensemble_pca,
    eval_ensemble_kfold,
    eval_ensemble_kfold_pca,
    eval_ensemble_combinatorics,
    eval_ensemble_combinatorics_pca,
)

PERSISTENT_PIPELINE_STEP_FILE = "persistent_pipeline_step.txt"

PIPELINE_STEPS = [
    download_models,
    extract_features,
    generate_visualizations,
    eval_ensemble,
    eval_ensemble_pca,
    eval_ensemble_kfold,
    eval_ensemble_kfold_pca,
    eval_ensemble_combinatorics,
    eval_ensemble_combinatorics_pca,
]

PIPELINE_STEPS_DINO_ONLY = [
    extract_dino_features,
    generate_visualizations,
    eval_ensemble,
    eval_ensemble_pca,
    eval_ensemble_kfold,
    eval_ensemble_kfold_pca,
    eval_ensemble_combinatorics,
    eval_ensemble_combinatorics_pca,
]
os.makedirs("emissions_logs", exist_ok=True)
@track_emissions(
        project_name="EISP on RCPD",
        output_dir="emissions_logs",
        log_level="error",
)
def main():
    print("Starting pipeline for using eisp on rcpd dataset...")

    # Get pipeline step from file
    pipeline_step = 0
    if os.path.exists(PERSISTENT_PIPELINE_STEP_FILE):
        with open(PERSISTENT_PIPELINE_STEP_FILE, "r") as f:
            pipeline_step = int(f.read().strip())
        print(f"Resuming from pipeline step: {pipeline_step}")

    for step in range(pipeline_step, len(PIPELINE_STEPS)):
        print(f"Executing pipeline step {step}: {PIPELINE_STEPS[step].__name__}")
        PIPELINE_STEPS[step]()

        # Update the persistent pipeline step file
        with open(PERSISTENT_PIPELINE_STEP_FILE, "w") as f:
            f.write(str(step + 1))

    # Now using only dino as feature

    for step in range(
        max(pipeline_step, len(PIPELINE_STEPS)),
        len(PIPELINE_STEPS_DINO_ONLY) + len(PIPELINE_STEPS),
    ):
        step = step - len(PIPELINE_STEPS)
        print(
            f"Executing pipeline step {step+len(PIPELINE_STEPS)}: {PIPELINE_STEPS_DINO_ONLY[step].__name__}"
        )
        PIPELINE_STEPS_DINO_ONLY[step]()

        # Update the persistent pipeline step file
        with open(PERSISTENT_PIPELINE_STEP_FILE, "w") as f:
            f.write(str(step + len(PIPELINE_STEPS) + 1))

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
