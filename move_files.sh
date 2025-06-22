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

echo "Connection String found. Searching for blobs with prefix 'frame_' to move..."

# --- Step 3: Find, Copy, and Delete all 'frame_' blobs ---
az storage blob list \
    --container-name "$CONTAINER_NAME" \
    --connection-string "$CONNECTION_STRING" \
    --prefix "frame_" \
    --query "[].name" \
    -o tsv | while IFS= read -r BLOB; do

    if [ -n "$BLOB" ]; then
      DESTINATION_BLOB_PATH="$DESTINATION_FOLDER/$BLOB"
      echo "Processing '$BLOB'..."

      # --- Part A: Copy the blob to the destination ---
      echo "  -> Copying to '$DESTINATION_BLOB_PATH'"
      az storage blob copy start \
        --destination-blob "$DESTINATION_BLOB_PATH" \
        --destination-container "$CONTAINER_NAME" \
        --source-blob "$BLOB" \
        --source-container "$CONTAINER_NAME" \
        --connection-string "$CONNECTION_STRING" \
        -o none

      # --- Part B: Verify copy and then delete the original source blob ---
      # This loop waits until the new blob exists before deleting the old one.
      while true; do
        az storage blob exists \
          --container-name "$CONTAINER_NAME" \
          --name "$DESTINATION_BLOB_PATH" \
          --connection-string "$CONNECTION_STRING" \
          --query "exists" \
          -o tsv | grep -q "true" && break
        # Wait 1 second before checking again
        sleep 1
      done
      
      echo "  -> Deleting original '$BLOB'"
      az storage blob delete \
        --container-name "$CONTAINER_NAME" \
        --name "$BLOB" \
        --connection-string "$CONNECTION_STRING" \
        -o none

      echo "  -> Move complete for '$BLOB'"
    fi
done

echo "All move operations have been completed."