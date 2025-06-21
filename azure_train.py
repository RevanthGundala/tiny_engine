from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml.constants import AssetTypes

# 1. Connect to your workspace
subscription_id = "04128013-a299-4e77-973c-b806970e9478"
resource_group = "rgundal2-rg"
workspace = "tiny-engine"
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# 2. Define the training environment
custom_env = Environment(
    name="tiny-engine-docker-env-2",
    description="Custom environment built from a Dockerfile for tiny engine.",
    build=BuildContext(
        path="./environment",  # <--- CHANGED: Point to the clean directory
        dockerfile_path="Dockerfile.azure" # This path is relative to 'path'
    ) 
)

# 3. Configure the command to run your training script
job = command(
    code="./src/",
    # The command is simple, no decompression needed
    command="python train.py --metadata_input ${{inputs.metadata}} --frames_input ${{inputs.frames}}",
    inputs={
        "metadata": Input(type=AssetTypes.URI_FILE, path="gamelogs/metadata.csv"),
        # Point to the final dataset registered from your sync
        "frames": Input(path="azureml:final-frames-dataset:1"),
    },
    environment=custom_env,
    compute="tiny-engine-cluster",
    display_name="tiny-engine-training-job",
    experiment_name="tiny-engine-training-experiment"
)

# 4. Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Submitted job: {returned_job.name}")