#!/bin/bash

# Script to clean the current directory, keeping specific files and the 'src' directory.

# --- Configuration: Items to KEEP ---
KEEP_DIR="src"
KEEP_FILES=(
    "shrimp_cfmU.py"
    "CFMExp_cmd_generate.py"
    "cleanup.sh" # Add the script itself to the keep list if it's in the target dir
)
# --- End Configuration ---

echo "This script will attempt to clean the current directory: $(pwd)"
echo "The following items will be KEPT if they exist:"
echo "Directory: $KEEP_DIR"
echo "Files:"
for f in "${KEEP_FILES[@]}"; do
    echo "  - $f"
done
echo ""

read -p "Are you sure you want to proceed with deletion? (yes/NO): " CONFIRMATION
CONFIRMATION=${CONFIRMATION:-NO} # Default to NO if user just presses Enter

if [[ "$CONFIRMATION" != "yes" && "$CONFIRMATION" != "YES" ]]; then
    echo "Cleanup aborted by user."
    exit 1
fi

echo "Proceeding with cleanup..."

# Loop through all items in the current directory
for item in *; do
    # Flag to check if the current item should be kept
    should_keep=false

    # 1. Check if it's the directory to keep
    if [[ -d "$item" && "$item" == "$KEEP_DIR" ]]; then
        should_keep=true
    fi

    # 2. Check if it's one of the files to keep (if not already marked to keep)
    if ! $should_keep ; then
        for keep_file in "${KEEP_FILES[@]}"; do
            if [[ -f "$item" && "$item" == "$keep_file" ]]; then
                should_keep=true
                break # Found in keep list, no need to check further
            fi
        done
    fi

    # 3. If not marked to keep, delete it
    if ! $should_keep ; then
        if [[ -d "$item" ]]; then
            echo "Deleting directory: $item"
            rm -rf "$item"
        elif [[ -f "$item" ]]; then
            echo "Deleting file: $item"
            rm -f "$item"
        else
            # This might catch symlinks or other special file types
            # For this specific ls output, it's unlikely to be hit.
            echo "Skipping unknown item type (not deleting): $item"
        fi
    else
        echo "Keeping: $item"
    fi
done

echo "Cleanup complete."
