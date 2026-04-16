#!/bin/bash

# Change to the script's directory (to avoid path issues)
cd "$(dirname "$0")"

# Kill any previous PyDecNef python processes to avoid server-client connection problems
pkill -9 python

# Initialize main.py script
python main.py