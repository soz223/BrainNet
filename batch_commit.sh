#!/usr/bin/env bash
set -euo pipefail

############################################
# Configuration
############################################

# The approximate maximum total size (in MB) allowed per commit/push batch
LIMIT_MB=90
LIMIT_BYTES=$((LIMIT_MB * 1024 * 1024))

# Commit message prefix
COMMIT_MESSAGE_PREFIX="Batch commit"

# Dry-run mode
DRY_RUN=false

############################################
# Helper Functions
############################################

timestamp() {
  date +"[%F %T]"
}

# Gather file sizes in a single pass (faster than 40k wc calls)
# Output: "size_in_bytes <space> filename"
collect_file_sizes() {
  # For large sets of files, using `stat` or `ls -l` may be faster.
  # We'll do an AWK parse on `ls -l`.
  #
  # NOTE: If you have filenames with spaces or newlines, this is tricky,
  # but hopefully your repo doesn’t have those. If it does, you'd need a more robust approach.

  echo "Collecting file sizes..."
  ls -l "${all_files_array[@]}" 2>/dev/null \
    | awk 'NR>1 {print $5, $9}'  # columns: size, filename
}

# Show progress numerically (simple)
show_progress() {
  local current=$1
  local total=$2
  echo -ne "$(timestamp) Processed $current/$total files...\r"
  if [ "$current" -eq "$total" ]; then
    echo
  fi
}

commit_and_push_batch() {
  local folder="$1"
  shift
  local batch_files=("$@")

  if [ ${#batch_files[@]} -eq 0 ]; then
    return 0
  fi

  echo
  echo "$(timestamp) --> Committing ${#batch_files[@]} file(s) from folder '$folder'"

  if [ "$DRY_RUN" = "true" ]; then
    echo "  [Dry-run] Would run: git add + git commit + git push"
    printf "  [Dry-run] Files in this batch:\n"
    for f in "${batch_files[@]}"; do
      echo "    $f"
    done
    echo
  else
    git add "${batch_files[@]}"
    git commit -m "${COMMIT_MESSAGE_PREFIX} from folder '$folder'"
    # If your Git might prompt for credentials, ensure you have a credential helper or SSH keys.
    git push
  fi
}

############################################
# Main Script
############################################

echo "$(timestamp) Collecting uncommitted files (untracked + modified)..."
uncommitted_files="$(git ls-files --others --modified --exclude-standard)"

if [ -z "$uncommitted_files" ]; then
  echo "$(timestamp) No uncommitted files found. Exiting."
  exit 0
fi

# Convert list to array
IFS=$'\n' read -r -d '' -a all_files_array < <(printf '%s\0' "$uncommitted_files" && printf '\0')

total_uncommitted=${#all_files_array[@]}
echo "$(timestamp) Found $total_uncommitted uncommitted file(s)."

# Collect sizes of all uncommitted files at once
declare -A file_size_map
while IFS= read -r line; do
  # line format: "SIZE FILENAME"
  # We'll parse each side. Some lines might be blank if something went wrong.
  size="$(echo "$line" | awk '{print $1}')"
  fname="$(echo "$line" | awk '{print $2}')"
  if [[ -n "$size" && -n "$fname" ]]; then
    file_size_map["$fname"]="$size"
  fi
done < <(collect_file_sizes)

echo "$(timestamp) Building top-level folder list..."
# If a file has no '/', top-level "folder" is the file itself.
# We'll keep it as a separate item.
folders="$(echo "$uncommitted_files" | sed 's#^\([^/]*\)/.*$#\1#' | sort -u)"

echo "Folders / top-level items with uncommitted files:"
echo "$folders"
echo

# We'll keep track of how many files have been processed, for progress messages.
processed=0

# Start folder loop
for folder in $folders; do
  echo "$(timestamp) Processing folder/item: $folder"

  # If $folder is actually a file (no slash in path), let's handle those separately
  # We'll detect that by seeing if there exist any actual subpaths for it.
  folder_files="$(echo "$uncommitted_files" | grep "^$folder/")"

  # CASE 1: We found subpaths -> $folder really is a folder
  if [ -n "$folder_files" ]; then
    # Normal folder logic
    batch_files=()
    batch_size=0

    while IFS= read -r file; do
      # Just in case the file disappeared
      if [ ! -f "$file" ]; then
        ((processed++))
        show_progress "$processed" "$total_uncommitted"
        continue
      fi

      # Lookup size from the map
      local_size=${file_size_map["$file"]:-0}

      # If adding this file would exceed the limit, commit now
      if [ $((batch_size + local_size)) -gt "$LIMIT_BYTES" ]; then
        commit_and_push_batch "$folder" "${batch_files[@]}"
        batch_files=()
        batch_size=0
      fi

      # Add file to batch
      batch_files+=("$file")
      batch_size=$((batch_size + local_size))

      ((processed++))
      show_progress "$processed" "$total_uncommitted"

    done <<< "$folder_files"

    # Commit leftover files in this folder
    commit_and_push_batch "$folder" "${batch_files[@]}"

  else
    # CASE 2: $folder is actually just a single file with no slash
    # See if it exists in the file_size_map
    if [ -f "$folder" ]; then
      # We treat it like a single-file batch
      ((processed++))
      show_progress "$processed" "$total_uncommitted"

      file_size=${file_size_map["$folder"]:-0}
      if [ "$file_size" -gt "$LIMIT_BYTES" ]; then
        echo "Warning: Single file '$folder' exceeds limit ($LIMIT_MB MB)."
        # We can still attempt to commit/push if the host allows it, or skip:
        # commit_and_push_batch "$folder" "$folder"
        # Or skip it:
        echo "Skipping '$folder' because it exceeds limit."
      else
        # Commit and push right away
        commit_and_push_batch "$folder" "$folder"
      fi
    else
      # If we get here, $folder is not a file, or something else changed
      echo "$(timestamp) Skipping '$folder'—not found or not a file."
    fi
  fi

  echo "$(timestamp) Finished folder/item: $folder"
  echo
done

echo "$(timestamp) All done!"
echo "Processed $processed out of $total_uncommitted files in total."
