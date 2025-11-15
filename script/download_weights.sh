#!/bin/bash

# !!! IMPORTANT !!!
# Replace this URL with the DIRECT download link you found using the browser's developer tools.
# The direct link will be different from the original one.
DIRECT_WEIGHTS_URL="https://nas-greenbotics.isr.uc.pt/drive/d/s/xkN8AYuu7uiP9n4kp2Am1fUNFxE2dLaa/webapi/entry.cgi/SPVSoAP3D_iros24.zip?api=SYNO.SynologyDrive.Files&method=download&version=2&files=%5B%22id%3A808813212301568107%22%5D&force_download=true&json_error=true&_dc=1763228781438&sharing_token=%222gUW8RE0fD87RBjw2k1NILTzkpYnSPeG5I11AZDXU9IPrkiimea12.gkbwXTK4hfm5.2KW_oEbW8ymVRcYRc7MonYrPQw.rrerCpPWVcvnTMg7Rqudu3aP6TckkgZgBAUMleky.7nQBKXXUIAEiT3FSKSj1562drbYgCvIsnltVjpZa0DeTNpogTysaTEbB4FyvQw0is1tNak_mEznvO3lKlyBqqvt0S3Br65H8Hc5VLPjjdN4whsIvc%22"

# Define the output directory and the name for the downloaded zip file
OUTPUT_DIR="./checkpoints"

# if OUTPUT_DIR does not exist, create it

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Created directory: $OUTPUT_DIR"
fi

OUTPUT_FILENAME="weights.zip"
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_FILENAME"

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Download the weights using the direct URL and save it with a specific name
echo "Downloading weights..."
wget -O "$OUTPUT_PATH" --no-check-certificate "$DIRECT_WEIGHTS_URL"

# Check if the download was successful and then unzip
if [ -f "$OUTPUT_PATH" ]; then
    echo "Download complete. Unzipping weights into $OUTPUT_DIR..."
    unzip -o "$OUTPUT_PATH" -d "$OUTPUT_DIR" # -o overwrites existing files without asking
    
    # Optional: remove the zip file after extraction
    # echo "Removing zip file..."
    # rm "$OUTPUT_PATH"
    
    echo "Weights are ready in $OUTPUT_DIR"
else
    echo "Error: Download failed. File not found at $OUTPUT_PATH"
fi