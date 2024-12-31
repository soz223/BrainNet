#!/usr/bin/env bash

set -euo pipefail
trap 'echo "Script failed on line $LINENO with exit code $?" >&2' ERR

LIMIT_MB=90
LIMIT_BYTES=$((LIMIT_MB * 1024 * 1024))
COMMIT_MESSAGE_PREFIX="Batch commit"
DRY_RUN=true  # for testing

echo "Collecting uncommitted files..."
uncommitted_files="$(git ls-files --others --modified --exclude-standard)"
if [ -z "$uncommitted_files" ]; then
  echo "No uncommitted files found."
  exit 0
fi

# Convert list to array
IFS=$'\n' read -r -d '' -a all_files_array < <(printf '%s\0' "$uncommitted_files" && printf '\0')
total_files=${#all_files_array[@]}
echo "Found $total_files uncommitted file(s)."

# --- CHUNKED file-size collection
collect_file_sizes_chunked() {
  local chunk=()
  local chunk_count=0
  local chunk_limit=1000

  for f in "${all_files_array[@]}"; do
    chunk+=("$f")
    ((chunk_count++))
    if (( chunk_count >= chunk_limit )); then
      ls -l "${chunk[@]}" 2>/dev/null \
        | awk 'NR>1 {print $5, $9}'
      chunk=()
      chunk_count=0
    fi
  done

  # leftover
  if (( chunk_count > 0 )); then
    ls -l "${chunk[@]}" 2>/dev/null \
      | awk 'NR>1 {print $5, $9}'
  fi
}

declare -A file_size_map

echo "Gathering file sizes (chunked) ..."
# Read "size filename" lines from chunked ls -l
while IFS= read -r line; do
  size="$(echo "$line" | awk '{print $1}')"
  fname="$(echo "$line" | awk '{print $2}')"
  if [[ -n "$size" && -n "$fname" ]]; then
    file_size_map["$fname"]="$size"
  fi
done < <(collect_file_sizes_chunked)

folders=$(echo "$uncommitted_files" | sed 's#^\([^/]*\)/.*$#\1#' | sort -u)
echo "Top-level items:"
echo "$folders"
echo

processed=0

commit_and_push_batch() {
  local folder="$1"
  shift
  local batch_files=("$@")

  if [ ${#batch_files[@]} -eq 0 ]; then
    return
  fi
  echo
  echo "  Committing ${#batch_files[@]} files in '$folder' (DRY_RUN=$DRY_RUN)"

  if [ "$DRY_RUN" = "true" ]; then
    echo "  [Dry-run] Files:"
    printf "    %s\n" "${batch_files[@]}"
  else
    git add "${batch_files[@]}"
    git commit -m "${COMMIT_MESSAGE_PREFIX} for $folder"
    git push
  fi
}

for folder in $folders; do
  echo "=== Processing folder/item: $folder ==="
  folder_files="$(echo "$uncommitted_files" | grep "^$folder/")"

  # Check if it has subpaths. If not, treat as single file.
  if [ -z "$folder_files" ]; then
    echo "  [DEBUG] No subpaths found for $folder. Checking if it's just a file..."
    if [ -f "$folder" ]; then
      echo "  It's a single file. Size: ${file_size_map["$folder"]:-0}"
      ((processed++))
      echo "  Processed $processed/$total_files"

      # If the file is bigger than limit, skip or warn
      if [ "${file_size_map["$folder"]:-0}" -gt "$LIMIT_BYTES" ]; then
        echo "  [WARNING] Single file '$folder' exceeds limit. Skipping."
      else
        commit_and_push_batch "$folder" "$folder"
      fi
    else
      echo "  '$folder' not a file? Skipping."
    fi
    continue
  fi

  # If we have subpaths, treat folder normally
  batch_files=()
  batch_size=0

  # Debug print
  echo "  [DEBUG] Subpaths for $folder:"
  echo "$folder_files" | sed 's/^/    /'

  while IFS= read -r file; do
    if [ ! -f "$file" ]; then
      echo "  [DEBUG] File '$file' no longer exists, skipping..."
      ((processed++))
      echo "  Processed $processed/$total_files"
      continue
    fi
    local_size=${file_size_map["$file"]:-0}
    if (( batch_size + local_size > LIMIT_BYTES )); then
      commit_and_push_batch "$folder" "${batch_files[@]}"
      batch_files=()
      batch_size=0
    fi
    batch_files+=("$file")
    batch_size=$((batch_size + local_size))

    ((processed++))
    echo "  Processed $processed/$total_files"
  done <<< "$folder_files"

  # Commit leftover in this folder
  commit_and_push_batch "$folder" "${batch_files[@]}"

  echo "=== Done with $folder ==="
  echo
done

echo "All done! Processed $processed/$total_files total files."
