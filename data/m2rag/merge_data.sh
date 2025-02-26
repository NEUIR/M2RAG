#!/bin/bash

mkdir -p fact_verify/cand_images

# Traverse all cand_images_* directories and move their contents to the cand_images directory.
for dir in fact_verify/cand_images_*; do
  if [ -d "$dir" ]; then
    mv "$dir"/* fact_verify/cand_images/  # Move data
    rmdir "$dir"  # Delete empty directories
  fi
done