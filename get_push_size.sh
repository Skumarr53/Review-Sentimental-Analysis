#!/bin/bash

# Get the list of staged files
staged_files=$(git diff --name-only  )

# Initialize total size variable
total_size=0

# Loop through each file and get its size
for file in $staged_files; do
  if [ -f "$file" ]; then
    # Get the size of the file
    file_size=$(du -b "$file" | cut -f1)
    # Add the file size to the total size
    total_size=$((total_size + file_size))
  fi
done

# Convert total size to human-readable format
total_size_hr=$(numfmt --to=iec $total_size)

echo "Total size of changes to be pushed: $total_size_hr"
