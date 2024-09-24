#!/bin/bash

# Function to handle git add, commit, and push in batches
process_files() {
    base_path=$1    # Base directory path
    file_prefix=$2  # File prefix (e.g., associated_high_order_fc_dynamic)
    start=$3        # Starting number (e.g., 1)
    end=$4          # Ending number (e.g., 1305)

    # Counter for tracking the number of files processed in the current batch
    counter=0

    # Loop through the specified range of files
    for sub in $(seq $start $end); do
        # Construct the file path
        file="${base_path}${file_prefix}${sub}.pt"

        # Add the file to Git
        git add "$file"

        # Increment the counter
        counter=$((counter + 1))

        # If 50 files have been added, commit and push them
        if [ $counter -eq 50 ]; then
            # Commit the batch of 50 files
            git commit -m "Adding batch of 50 files ending at ${file_prefix}${sub}.pt"

            # Push the commit
            git push

            # Reset the counter for the next batch
            counter=0

            # Optional: Add a delay between each push to prevent overload (adjust the sleep time if needed)
            sleep 2
        fi
    done

    # Commit and push any remaining files if there are less than 50
    if [ $counter -gt 0 ]; then
        git commit -m "Adding final batch of files ending at ${file_prefix}${sub}.pt"
        git push
    fi
}

# # Call the function for the first case
# process_files "/data/soz223/BrainNet/data/ADNI/fmri_edge/associated_high_order_fc_dynamic/" "associated_high_order_fc_dynamic" 1 1305

# # Call the function for the second case
# process_files "/data/soz223/BrainNet/data/ADNI/fmri_edge/correlations_correlation_dynamic/" "correlations_correlation_dynamic" 1 1305

# # Call the function for the third case
# process_files "/data/soz223/BrainNet/data/ADNI/fmri_edge/euclidean_distance_dynamic/" "euclidean_distance_dynamic" 1 726

# process_files "/data/soz223/BrainNet/data/ADNI/fmri_edge/spearman_correlation_dynamic/" "spearman_correlation_dynamic" 1 1305

# process_files "/data/soz223/BrainNet/data/ADNI/fmri_edge/knn_graph_dynamic/" "knn_graph_dynamic" 1 1305

# process_files "/data/soz223/BrainNet/data/ADNI/fmri_edge/kendall_correlation_dynamic/" "kendall_correlation_dynamic" 371 1305

# process_files "/data/soz223/BrainNet/data/ADNI/fmri_edge/mutual_information_dynamic/" "mutual_information_dynamic" 1 111