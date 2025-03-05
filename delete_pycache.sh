#!/bin/bash

# This script deletes all __pycache__ directories in the current directory and its subdirectories.

# Use the find command to locate and delete all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Use the find command to locate and delete all .cache directories
find . -type d -name ".cache" -exec rm -rf {} +

# Print a message indicating completion
echo "All __pycache__ and .cache directories have been deleted."
