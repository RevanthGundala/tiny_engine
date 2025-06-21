#!/bin/bash

# Step 1: Set variables
STORAGE_ACCOUNT_NAME="tinyengine7049014480"
RESOURCE_GROUP="rgundal2-rg"
CONTAINER_NAME="azureml-blobstore-39931bb3-7248-4e33-acab-73883295a3b5"
DESTINATION_FOLDER="frames"

# Step 2: Get the Storage Connection String
echo "Fetching storage connection string..."
CONNECTION_STRING=$(az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "connectionString" \
  -o tsv)

if [ -z "$CONNECTION_STRING" ]; then
  echo "Error: Could not fetch connection string."
  exit 1
fi

echo "Connection String found. Searching for blobs with prefix 'frame_'..."

# --- Step 3: Find and Move all 'frame_' blobs using the --prefix parameter ---
# This is the most efficient and standard way to list files with a common prefix.
az storage blob list \
    --container-name "$CONTAINER_NAME" \
    --connection-string "$CONNECTION_STRING" \
    --prefix "frame_" \
    --query "[].name" \
    -o tsv | while IFS= read -r BLOB; do

    if [ -n "$BLOB" ]; then
      echo "Initiating move for '$BLOB' to '$DESTINATION_FOLDER/'"
      az storage blob move start \
        --destination-blob "$DESTINATION_FOLDER/$BLOB" \
        --destination-container "$CONTAINER_NAME" \
        --source-blob "$BLOB" \
        --source-container "$CONTAINER_NAME" \
        --connection-string "$CONNECTION_STRING" \
        -o none
    fi
done

echo "All move operations have been initiated successfully."
echo "Note: The move happens asynchronously in Azure."